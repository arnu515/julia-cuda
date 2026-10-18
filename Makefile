# Builds the kernels (one PTX file per GPU arch), the host binary that embeds them,
# and a copy of that binary that runs outside nix. Must be run in the nix shell.
#
#   make               PTX (every tier) + release host binary embedding tier ARCH
#   make dist          dist/julia-cuda-sm_ARCH: stripped, loader at /lib64, for the GPU machines
#   make dist ARCH=89  the same for another tier: 75 (2080 Ti, default), 80, 89 (4070), 120
#   make CHECKED=1     PTX with slice bounds checks (an out-of-bounds index traps)
#   make -B ptx VERBOSE=1   also print ptxas's register / spill / shared memory report

TIERS := 75 80 89 120
ARCH ?= 75
ifeq ($(filter $(ARCH),$(TIERS)),)
$(error ARCH=$(ARCH) is not one of: $(TIERS))
endif
FEATURES_75 :=
FEATURES_80 := sm_80
FEATURES_89 := sm_80,sm_89
FEATURES_120 := sm_80,sm_89,sm_120

PTX := $(TIERS:%=ptx/sm_%/kernels.ptx)
KERNEL_SRC := $(sort $(shell find kernels/src -name '*.rs')) kernels/Cargo.toml kernels/Cargo.lock
# cargo oxide rewrites src/main.rs on every build (to force a recompile), so we use a hash instead.
SRC_HASH := $(shell { cat $(KERNEL_SRC); echo "CHECKED=$(CHECKED)"; } | sha256sum | cut -c1-16)
STAMP := ptx/.src-$(SRC_HASH)
# Recursive (=), so cargo is only asked when dist needs it. Not ./target: cargo's config may
# put it elsewhere.
TARGET_DIR = $(shell cargo metadata --format-version 1 --no-deps | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)

.PHONY: build ptx dist clean
# Tiers can't build at the same time: they share cargo's target directory (and its lock), and
# cargo oxide rewrites src/main.rs during each build, which another tier's rustc may read half-written.
.NOTPARALLEL:
# A tier whose ptxas check fails must not be left behind looking up to date.
.DELETE_ON_ERROR:

build: ptx
	cargo build --release --features sm_$(ARCH)

ptx: $(PTX)

$(STAMP):
	@mkdir -p ptx
	@rm -f ptx/.src-*
	@touch $@

# CUDA_OXIDE_PTX_DIR puts the PTX (and the .ll files beside it) there instead of in kernels/.
ptx/sm_%/kernels.ptx: $(STAMP)
	cd kernels && CUDA_OXIDE_PTX_DIR=$(CURDIR)/ptx/sm_$* cargo oxide build --arch sm_$* $(if $(FEATURES_$*),--features $(FEATURES_$*)) \
		$(if $(CHECKED),,--unchecked-indexing)
	ptxas -arch=sm_$* $(if $(VERBOSE),-v) $@ -o /dev/null
	@echo "$@: $$(grep -m1 '^\.version' $@), $$(grep -c '^\.visible \.entry' $@) kernels"

dist: build
	@mkdir -p dist
	cp $(TARGET_DIR)/release/julia-cuda dist/julia-cuda-sm_$(ARCH)
	strip dist/julia-cuda-sm_$(ARCH)
	patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath dist/julia-cuda-sm_$(ARCH)

clean:
	rm -rf ptx dist
