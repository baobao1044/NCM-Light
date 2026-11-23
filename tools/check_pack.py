"""
Check the NCM-Light shader pack before it ever reaches the game.

Checks performed:
  1. structure  -- every program across shaders/, world-1/, world1/ has both
                   .vsh and .fsh stages, every composite/gbuffers fragment
                   shader declares /* DRAWBUFFERS:... */ directly in the .fsh,
                   every stage starts with #version 120, includes resolve,
                   options & lang keys are consistent, buffer sizes per pass
                   have matching viewports, and ping-pong flips are not broken.
  2. glsl       -- expand OptiFine/Iris #include directives across ALL
                   dimension folders and compile every .vsh/.fsh with
                   glslangValidator (also testing DEBUG_LEVEL=2..5 & SHADOWS=0).

Usage:
    python tools/check_pack.py <pack-root> [--glslang <path to glslangValidator>]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

INCLUDE_RE = re.compile(r'^\s*#include\s+"?([^"<\s]+)"?\s*$')
OPTION_RE = re.compile(r'^#define\s+([A-Z0-9_]+)\s+([^/\s]+)\s*//')
CONST_OPTION_RE = re.compile(r'^const\s+(?:int|float)\s+([A-Za-z0-9_]+)\s*=\s*([^;]+);\s*//\s*\[')
BUFFER_FORMAT_RE = re.compile(r'\bconst\s+\w+\s+\w*Format\s*=')
SIZE_RE = re.compile(r'^\s*size\.buffer\.(colortex\d+)\s*=\s*([0-9.]+)\s+([0-9.]+)\s*$')
FLIP_FALSE_RE = re.compile(r'^\s*flip\.(\w+)\.(colortex\d+)\s*=\s*false\s*$', re.IGNORECASE)
DRAWBUFFERS_RE = re.compile(r'/\*\s*DRAWBUFFERS:([0-9]+)\s*\*/')
PER_BUFFER_BLEND_RE = re.compile(r'^\s*blend\.\w+\.colortex\d+\s*=')


def strip_comments(line, in_block):
    """Return (code on this line, whether a /* */ block is still open)."""
    code = ""
    index = 0
    while index < len(line):
        if in_block:
            end = line.find("*/", index)
            if end < 0:
                return code, True
            in_block = False
            index = end + 2
            continue
        block = line.find("/*", index)
        line_comment = line.find("//", index)
        if line_comment >= 0 and (block < 0 or line_comment < block):
            return code + line[index:line_comment], False
        if block >= 0:
            code += line[index:block]
            in_block = True
            index = block + 2
            continue
        return code + line[index:], False
    return code, in_block


def shader_programs_in_dir(target_dir):
    """Return {program_name: {"vsh": path, "fsh": path}} for a dimension directory."""
    programs = {}
    if not os.path.isdir(target_dir):
        return programs
    for name in sorted(os.listdir(target_dir)):
        full = os.path.join(target_dir, name)
        if not os.path.isfile(full):
            continue
        stem, ext = os.path.splitext(name)
        if ext not in (".vsh", ".fsh"):
            continue
        if not (stem.startswith("composite") or stem.startswith("gbuffers_")
                or stem == "final" or stem == "shadow"):
            continue
        programs.setdefault(stem, {})[ext.lstrip(".")] = full
    return programs


def all_dimension_programs(shaders_dir):
    """Return [(dim_label, dim_dir, programs_dict)]."""
    dims = [("overworld", shaders_dir, shader_programs_in_dir(shaders_dir))]
    for sub, label in (("world-1", "nether"), ("world1", "the_end")):
        sub_dir = os.path.join(shaders_dir, sub)
        if os.path.isdir(sub_dir):
            dims.append((label, sub_dir, shader_programs_in_dir(sub_dir)))
    return dims


def expand_includes(path, shaders_dir, seen=None):
    """Inline #include "/x.glsl" relative to shaders_dir (or local dir if relative)."""
    seen = seen or set()
    if path in seen:
        raise RuntimeError("circular include: %s" % path)
    seen = seen | {path}

    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = INCLUDE_RE.match(line)
            if not match:
                out.append(line)
                continue
            inc = match.group(1)
            if inc.startswith(("/", "\\")):
                target = os.path.join(shaders_dir, inc.lstrip("/\\"))
            else:
                target = os.path.join(os.path.dirname(path), inc)
            target = os.path.normpath(target)
            if not os.path.isfile(target):
                raise RuntimeError("%s: cannot resolve %s" % (path, inc))
            out.append(expand_includes(target, shaders_dir, seen))
    return "".join(out)


def loader_directive_checks(source, name, stage):
    """Reject bare buffer format directives outside /* */ comments, duplicate
    #version directives, and prose comments that hijack Iris's lastIndexOf
    DRAWBUFFERS/RENDERTARGETS parser!"""
    problems = []
    in_block = False
    version_lines = []
    for number, line in enumerate(source.splitlines(), start=1):
        code, in_block = strip_comments(line, in_block)
        if code.strip().startswith("#version"):
            version_lines.append(number)
        if BUFFER_FORMAT_RE.search(code):
            problems.append("%s:%d: buffer format directive outside a comment"
                            " block (would fail to compile): %s"
                            % (name, number, line.strip()))
    if len(version_lines) != 1 or version_lines[0] != 1:
        problems.append("%s: expected exactly one #version on line 1, got lines %s"
                        % (name, version_lines))

    # Mirror Iris CommentDirectiveParser.findDirective() which uses lastIndexOf!
    if stage == "fsh":
        db_count = source.count("DRAWBUFFERS:")
        rt_count = source.count("RENDERTARGETS:")
        if rt_count > 0:
            problems.append("%s.fsh: unexpected RENDERTARGETS: token in expanded source" % name)
        if db_count > 1:
            problems.append("%s.fsh: %d occurrences of DRAWBUFFERS: in expanded source"
                            " (prose comment collision with Iris lastIndexOf!)" % (name, db_count))
        if db_count >= 1:
            idx = source.rfind("DRAWBUFFERS:")
            end = source.find("*/", idx)
            if end < 0:
                problems.append("%s.fsh: unclosed DRAWBUFFERS comment" % name)
            else:
                payload = source[idx + len("DRAWBUFFERS:"):end].strip()
                if not payload.isdigit():
                    problems.append("%s.fsh: Iris lastIndexOf DRAWBUFFERS payload is %r"
                                    " (must be digits only!)" % (name, payload))
                else:
                    frag_indices = {int(m) for m in re.findall(r'gl_FragData\[\s*(\d+)\s*\]', source)}
                    expected_indices = set(range(len(payload)))
                    if frag_indices != expected_indices:
                        problems.append("%s.fsh: gl_FragData indices %s do not match"
                                        " DRAWBUFFERS:%s (%d attachment(s))"
                                        % (name, sorted(frag_indices), payload, len(payload)))
    return problems


def properties_and_lang_checks(shaders_dir, overworld_programs):
    problems = []
    props = os.path.join(shaders_dir, "shaders.properties")
    screen_opts = []
    profiles = {}
    written_by_pass = {}

    for pname, stages in overworld_programs.items():
        fsh = stages.get("fsh")
        if fsh:
            with open(fsh, "r", encoding="utf-8") as handle:
                m = DRAWBUFFERS_RE.search(handle.read())
                if m:
                    written_by_pass[pname] = {"colortex%s" % c for c in m.group(1)}

    if not os.path.isfile(props):
        problems.append("shaders.properties: missing")
        return problems

    with open(props, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if PER_BUFFER_BLEND_RE.match(line):
                problems.append("shaders.properties:%d: per-buffer blend (%s)"
                                " requires GL 4.0 / ARB_draw_buffers_blend and"
                                " fails on OpenGL 3.3 Core GPUs" % (lineno, line))
            m_flip = FLIP_FALSE_RE.match(line)
            if m_flip:
                prog, buf = m_flip.group(1), m_flip.group(2)
                if buf in written_by_pass.get(prog, set()):
                    problems.append("shaders.properties:%d: %s disables ping-pong"
                                    " flip on %s written by %s, leaving readers on"
                                    " the unwritten buffer!" % (lineno, line, buf, prog))
            if line.startswith("screen"):
                _, rhs = line.split("=", 1)
                screen_opts = [tok for tok in rhs.split() if tok != "<empty>"]
            elif line.startswith("profile."):
                pname, rhs = line.split("=", 1)
                profiles[pname.strip()] = rhs.strip().split()

    declared = {}
    for opt_file in ("options.glsl", "common.glsl"):
        opath = os.path.join(shaders_dir, opt_file)
        if os.path.isfile(opath):
            with open(opath, "r", encoding="utf-8") as handle:
                for line in handle:
                    m1 = OPTION_RE.match(line)
                    if m1:
                        opt, val = m1.group(1), m1.group(2)
                        if opt in declared:
                            problems.append("option %s declared twice" % opt)
                        declared[opt] = val
                    m2 = CONST_OPTION_RE.match(line)
                    if m2:
                        opt, val = m2.group(1), m2.group(2)
                        declared[opt] = val

    if not declared:
        problems.append("options.glsl / common.glsl: no options declared")
    else:
        for opt in screen_opts:
            if opt not in declared:
                problems.append("shaders.properties screen option %r not declared in options.glsl/common.glsl" % opt)
        for pname, assigns in profiles.items():
            for tok in assigns:
                if "=" in tok:
                    k = tok.split("=", 1)[0].strip()
                    if k not in declared:
                        problems.append("shaders.properties %s references undeclared option %r" % (pname, k))

    for lang_file in ("en_US.lang", "vi_VN.lang"):
        lpath = os.path.join(shaders_dir, "lang", lang_file)
        if not os.path.isfile(lpath):
            problems.append("lang/%s: missing" % lang_file)
            continue
        with open(lpath, "r", encoding="utf-8") as handle:
            keys = {ln.split("=", 1)[0].strip() for ln in handle if "=" in ln}
        for opt in screen_opts:
            if ("option.%s" % opt) not in keys:
                problems.append("lang/%s: missing label option.%s" % (lang_file, opt))
        for pname in profiles:
            if pname not in keys:
                problems.append("lang/%s: missing label %s" % (lang_file, pname))

    return problems


def structural_and_size_checks(shaders_dir, dim_sets):
    problems = []
    sizes = {}
    props = os.path.join(shaders_dir, "shaders.properties")
    if os.path.isfile(props):
        with open(props, "r", encoding="utf-8") as handle:
            for line in handle:
                match = SIZE_RE.match(line)
                if match:
                    sizes[int(match.group(1).replace("colortex", ""))] = (
                        float(match.group(2)), float(match.group(3)))

    required_gbuffers = {
        "gbuffers_basic", "gbuffers_skybasic", "gbuffers_skytextured",
        "gbuffers_textured", "gbuffers_textured_lit", "gbuffers_terrain",
        "gbuffers_block", "gbuffers_entities", "gbuffers_water", "gbuffers_hand",
    }

    for dim_label, _dim_dir, programs in dim_sets:
        missing_gb = required_gbuffers - set(programs.keys())
        if missing_gb:
            problems.append("[%s] missing required gbuffers programs: %s"
                            % (dim_label, ", ".join(sorted(missing_gb))))
        for name, stages in programs.items():
            for stage in ("vsh", "fsh"):
                path = stages.get(stage)
                if not path:
                    problems.append("[%s] %s: missing .%s" % (dim_label, name, stage))
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                first_line = content.splitlines()[0].strip() if content.splitlines() else ""
                if first_line != "#version 120":
                    problems.append("[%s] %s.%s: first line is %r, expected #version 120"
                                    % (dim_label, name, stage, first_line))
                if stage == "fsh" and (name.startswith("composite") or name.startswith("gbuffers_")):
                    match = DRAWBUFFERS_RE.search(content)
                    if not match:
                        problems.append("[%s] %s.fsh: missing /* DRAWBUFFERS:n */ in .fsh file"
                                        % (dim_label, name))
                    else:
                        written = [int(ch) for ch in match.group(1)]
                        scales = {sizes.get(buf, (1.0, 1.0)) for buf in written}
                        if len(scales) > 1:
                            problems.append("[%s] %s.fsh: DRAWBUFFERS writes mixed sizes %s"
                                            % (dim_label, name, written))
    return problems


def compile_source(glslang, workdir, tag, stage, source):
    out = os.path.join(workdir, "%s.%s" % (tag, stage))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(source)
    result = subprocess.run(
        [glslang, "-S", "vert" if stage == "vsh" else "frag", out],
        capture_output=True, text=True)
    if result.returncode != 0:
        return "%s.%s:\n%s" % (tag, stage, (result.stdout + result.stderr).strip())
    return None


def glsl_checks(shaders_dir, dim_sets, glslang, workdir):
    problems = []
    checked = 0
    for dim_label, _dim_dir, programs in dim_sets:
        for name, stages in programs.items():
            for stage in ("vsh", "fsh"):
                path = stages.get(stage)
                if not path:
                    continue
                tag = "%s__%s" % (dim_label, name)
                try:
                    source = expand_includes(path, shaders_dir)
                except RuntimeError as exc:
                    problems.append(str(exc))
                    continue
                problems += loader_directive_checks(source, tag, stage)
                err = compile_source(glslang, workdir, tag, stage, source)
                checked += 1
                if err:
                    problems.append(err)

                # Also exercise key option permutations on overworld composites/final/terrain/shadow
                if dim_label == "overworld" and stage == "fsh" and name in ("composite1", "composite2", "final"):
                    variants = [
                        ("shadows0",  source.replace("#define SHADOWS 1", "#define SHADOWS 0")),
                        ("rtgi0",     source.replace("#define RT_GI 1", "#define RT_GI 0")),
                        ("rtrefl0",   source.replace("#define RT_REFLECTIONS 1", "#define RT_REFLECTIONS 0")),
                        ("clouds0",   source.replace("#define VOLUMETRIC_CLOUDS 1", "#define VOLUMETRIC_CLOUDS 0")),
                        ("cshadow0",  source.replace("#define COLORED_SHADOWS 1", "#define COLORED_SHADOWS 0")),
                        ("caustics0", source.replace("#define WATER_CAUSTICS 1", "#define WATER_CAUSTICS 0")),
                        ("dbg2",      source.replace("#define DEBUG_LEVEL 0", "#define DEBUG_LEVEL 2")),
                        ("dbg3",      source.replace("#define DEBUG_LEVEL 0", "#define DEBUG_LEVEL 3")),
                        ("dbg4",      source.replace("#define DEBUG_LEVEL 0", "#define DEBUG_LEVEL 4")),
                        ("dbg5",      source.replace("#define DEBUG_LEVEL 0", "#define DEBUG_LEVEL 5")),
                        ("dbg6",      source.replace("#define DEBUG_LEVEL 0", "#define DEBUG_LEVEL 6")),
                    ]
                    for vname, vsrc in variants:
                        err = compile_source(glslang, workdir, "%s_%s" % (tag, vname), stage, vsrc)
                        checked += 1
                        if err:
                            problems.append(err)
                elif dim_label == "overworld" and name in ("gbuffers_terrain", "shadow"):
                    off_macro = ("#define WAVING_FOLIAGE 1", "#define WAVING_FOLIAGE 0") if stage == "vsh" else ("#define LABPBR_SUPPORT 1", "#define LABPBR_SUPPORT 0")
                    if off_macro[0] in source:
                        err = compile_source(glslang, workdir, "%s_off" % tag, stage, source.replace(off_macro[0], off_macro[1]))
                        checked += 1
                        if err:
                            problems.append(err)
    return problems, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack_root")
    parser.add_argument("--glslang")
    args = parser.parse_args()

    shaders_dir = os.path.join(args.pack_root, "shaders")
    if not os.path.isdir(shaders_dir):
        print("no shaders/ directory in %s" % args.pack_root)
        return 1

    dim_sets = all_dimension_programs(shaders_dir)
    for dim_label, _dim_dir, progs in dim_sets:
        print("[%s] %d programs: %s" % (dim_label, len(progs), ", ".join(sorted(progs))))

    problems = []
    problems += properties_and_lang_checks(shaders_dir, dim_sets[0][2])
    problems += structural_and_size_checks(shaders_dir, dim_sets)
    print("structure & properties: %d problem(s)" % len(problems))

    if args.glslang and os.path.isfile(args.glslang):
        workdir = tempfile.mkdtemp(prefix="ncm-prebuild-")
        glsl_problems, checked = glsl_checks(shaders_dir, dim_sets, args.glslang, workdir)
        shutil.rmtree(workdir, ignore_errors=True)
        print("glsl: %d shader permutation(s) compiled, %d problem(s)" % (checked, len(glsl_problems)))
        problems += glsl_problems
    else:
        print("glsl: skipped (no glslangValidator given)")

    for problem in problems:
        print("\n---\n%s" % problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
