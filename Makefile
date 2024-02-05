.PHONY: all
all: build

#
# Test
#
.PHONY: test
test:

#
# Build
#
yolov8_rknn: install/examples/yolov8/model/yolov8_rk3588.rknn install/examples/yolov8/model/yolov8_rk3568.rknn install/examples/yolov8/model/yolov8_rk3566.rknn
examples/yolov8/model/yolov8n.onnx:
	cd ./examples/yolov8/model && \
	bash ./download_model.sh

install/examples/yolov8/model/yolov8_rk3588.rknn: examples/yolov8/model/yolov8n.onnx
	cd ./examples/yolov8/python && \
	python3 convert.py ../model/$(notdir $<) rk3588 i8 ../../../$@

install/examples/yolov8/model/yolov8_rk3568.rknn: examples/yolov8/model/yolov8n.onnx
	cd ./examples/yolov8/python && \
	python3 convert.py ../model/$(notdir $<) rk3568 i8 ../../../$@

install/examples/yolov8/model/yolov8_rk3566.rknn: examples/yolov8/model/yolov8n.onnx
	cd ./examples/yolov8/python && \
	python3 convert.py ../model/$(notdir $<) rk3566 i8 ../../../$@

install/examples/yolov8/model:
	mkdir -p $@

.PHONY: build
build: install/examples/yolov8/model yolov8_rknn

#
# Clean
#
.PHONY: distclean
distclean: clean

clean: clean-deb
	find ./ -type f -name "*.onnx" -exec rm {} \;
	find ./ -type f -name "*.rknn" -exec rm {} \;
	rm -rf install/examples

.PHONY: clean-deb
clean-deb:
	rm -rf debian/.debhelper debian/rknn-model-zoo-rk3566 debian/rknn-model-zoo-rk3568 debian/rknn-model-zoo-rk3588 debian/debhelper-build-stamp debian/files debian/*.debhelper.log debian/*.postrm.debhelper debian/*.substvars debian/*.debhelper.log

#
# Release
#
.PHONY: dch
dch: debian/changelog
	EDITOR=true gbp dch --commit --debian-branch=main --release --dch-opt=--upstream

.PHONY: deb
deb: debian
	debuild --no-lintian --lintian-hook "lintian --fail-on error,warning --suppress-tags bad-distribution-in-changes-file -- %p_%v_*.changes" --no-sign -b

.PHONY: release
release:
	gh workflow run .github/workflows/new_version.yml
