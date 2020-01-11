.PHONY: clean

all: compute_simple.comp.spv compute_acc.comp.spv compute_pos.comp.spv shader_base.frag.spv shader_base.vert.spv

%.comp.spv: %.comp
	glslangValidator  -V $< -o $@

%.frag.spv: %.frag
	glslangValidator  -V $< -o $@

%.vert.spv: %.vert
	glslangValidator  -V $< -o $@

clean:
	-rm *.spv



