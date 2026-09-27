#!/usr/bin/env python3
"""Apply the verified geometry-width patch to a generated v2 engine snapshot.

The transform matches exact setup and bounds-check expressions, independent
of feature-parameter values or the snapshot's overall SHA. It refuses a
candidate whose malformed-descriptor guards changed, and it never edits the
input snapshot. Run RTL regression and Gowin routing on each resulting image;
the all-exact proof does not certify unrelated changes in another snapshot.
"""
import argparse
import hashlib
from pathlib import Path


COMMENT_OLD = '// Arithmetic, model data, descriptors and the operating clock are unchanged.'
COMMENT_NEW = ("// Experimental geometry width proof. Descriptor arithmetic and output values\n"
               "// are unchanged for every valid ABI descriptor; existing full-width checks\n"
               "// reject malformed fields before any shortened product can be used.")

REPLACEMENTS = (
    ("    wire [31:0] plane_calc=32'(ih)*32'(iw);",
     "    // Legal dimensions: ih,iw<=255; kh,kw<=31; ic,oc<=1024;\n"
     "    // stride<=32768; and each output axis is at most 285. GEOM_CHECK keeps\n"
     "    // the original full-field rejection predicates for larger encodings.\n"
     "    wire [15:0] plane_calc=ih[7:0]*iw[7:0];"),
    ("    wire [31:0] ow_calc=(sw==2)?((numerator_w>>1)+1):((sw==1)?(numerator_w+1):1);",
     "    wire [31:0] ow_calc=(sw==2)?((numerator_w>>1)+1):((sw==1)?(numerator_w+1):1);\n"
     "    wire [12:0] row_step_calc=sh[4:0]*iw[7:0];\n"
     "    wire [12:0] pad_offset_calc=pt[4:0]*iw[7:0];\n"
     "    wire [9:0] kernel_area_calc=kh[4:0]*kw[4:0];\n"
     "    wire [17:0] output_area_calc=oh_calc[8:0]*ow_calc[8:0];\n"
     "    wire [26:0] input_bytes_calc=plane[15:0]*ic[10:0];\n"
     "    wire [20:0] reduction_calc=kernel_area[9:0]*ic[10:0];\n"
     "    wire [28:0] outputs_calc=output_area[17:0]*oc_total[10:0];\n"
     "    wire [31:0] gemm_weight_span=outputs[15:0]*stride[15:0];\n"
     "    wire [26:0] spatial_weight_span=oc_total[10:0]*stride[15:0];"),
    ("{32'b0,wb}+{32'b0,outputs}*{32'b0,stride}>64'(MEM_BYTES)",
     "{32'b0,wb}+{32'b0,gemm_weight_span}>64'(MEM_BYTES)"),
    ("                    plane<=plane_calc;row_step<=32'(sh)*32'(iw);pad_offset<=32'(pt)*32'(iw);\n"
     "                    kernel_area<=32'(kh)*32'(kw);output_area<=oh_calc*ow_calc;",
     "                    plane<=plane_calc;row_step<=row_step_calc;pad_offset<=pad_offset_calc;\n"
     "                    kernel_area<=kernel_area_calc;output_area<=output_area_calc;"),
    ("                    input_bytes_reg<=plane*32'(ic);\n"
     "                    reduction_reg<=(op==OP_DWCONV)?kernel_area:kernel_area*32'(ic);\n"
     "                    outputs_reg<=output_area*32'(oc_total);",
     "                    input_bytes_reg<=input_bytes_calc;\n"
     "                    reduction_reg<=(op==OP_DWCONV)?kernel_area:reduction_calc;\n"
     "                    outputs_reg<=outputs_calc;"),
    ("{32'b0,wb}+{48'b0,oc_total}*{32'b0,stride}>64'(MEM_BYTES)",
     "{32'b0,wb}+{37'b0,spatial_weight_span}>64'(MEM_BYTES)"),
)

# These full-field predicates are the safety gate for every narrowed operand.
# A changed guard requires a new proof and must make this transform fail.
GENERAL_GUARDS = (
    "else if(count==0||outputs==0||count>MEM_BYTES||outputs>MEM_BYTES",
    "else state<=spatial?GEOM0:OTHER_CHECK;",
)
GEOMETRY_GUARDS = (
    "kh==0||kw==0||kh>31||kw>31||ih==0||iw==0||ic==0||oc_total==0||",
    "ih>255||iw>255||ic>1024||oc_total>1024||",
    "pt>=kh||pbm>=kh||pl>=kw||pr>=kw||",
    "32'(ih)+32'(pt)+32'(pbm)<32'(kh)||32'(iw)+32'(pl)+32'(pr)<32'(kw)||",
    "stride<count||stride>MEM_BYTES||",
)


def verify_rejection_guards(source):
    """Ensure raw fields still reject every value outside narrowed bounds."""
    if source.count('module v2_engine #(') != 1:
        raise ValueError('expected one v2_engine module')
    begin = source.find('D_CHECK:begin')
    geometry = source.find('GEOM_CHECK:begin')
    finish = source.find('LINE_CLEAR:begin', geometry)
    if not (0 <= begin < geometry < finish):
        raise ValueError('descriptor check sequence changed')
    for guard in GENERAL_GUARDS:
        if source[begin:geometry].count(guard) != 1:
            raise ValueError(f'general descriptor guard changed: {guard}')
    for guard in GEOMETRY_GUARDS:
        if source[geometry:finish].count(guard) != 1:
            raise ValueError(f'geometry descriptor guard changed: {guard}')
    if source[begin:geometry].count('stride<count||stride>MEM_BYTES||') != 1:
        raise ValueError('GEMM stride guard changed')


def apply_geometry_narrowing(source):
    """Return a patched snapshot; reject missing/ambiguous arithmetic sites."""
    verify_rejection_guards(source)
    if 'wire [15:0] plane_calc=ih[7:0]*iw[7:0];' in source:
        raise ValueError('geometry patch already applied')
    result = source
    if COMMENT_OLD in result:
        result = result.replace(COMMENT_OLD, COMMENT_NEW, 1)
    for old, new in REPLACEMENTS:
        if result.count(old) != 1:
            raise ValueError(f'geometry arithmetic site changed: {old}')
        result = result.replace(old, new, 1)
    verify_rejection_guards(result)
    return result


def materialize(source_path, output_path):
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError('source and output must differ')
    original = source_path.read_text()
    patched = apply_geometry_narrowing(original)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_text() != patched:
            raise FileExistsError('existing generated snapshot differs')
    else:
        output_path.write_text(patched)
    digest = lambda data: hashlib.sha256(data.encode()).hexdigest()
    return dict(source=str(source_path), output=str(output_path),
                input_sha256=digest(original), output_sha256=digest(patched),
                verified_guards=len(GENERAL_GUARDS)+len(GEOMETRY_GUARDS)+1,
                exact_replacements=len(REPLACEMENTS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(materialize(args.source, args.output), indent=2))
