"""test-compri.py

一次性对比运行 src/test.py 中的四种开关组合并保存输出：
 1) 纯 YOLO
 2) MSICN 开，FIE 关
 3) MSICN 关，FIE 开
 4) 全开

脚本会逐个加载 src/test.py 的代码（以独立 globals 执行），在每次调用前设置
`ENABLE_MSICN` / `ENABLE_FIE` 两个全局变量，从而复用 `run_single_image_pipeline()`。
运行完成后，会列出 results/single_image_pipeline/pipeline_visualization 下新增的文件。

路径与权重配置说明：
 1. 本脚本不单独维护图片路径和 checkpoint。
 2. 它完全复用 src/test.py 顶部宏定义中的 TARGET_IMAGE_NAME、YOLO_WEIGHTS_PATH、
	 ICFIE_CHECKPOINT_PATH、YOLO_ONLY_CHECKPOINT_PATH。
 3. 因此要运行 test-compri，先把 src/test.py 顶部配置改对，再执行本脚本。
"""

from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
TEST_PY = ROOT / "test.py"


def _exec_test_with_flags(enable_msicn: bool, enable_fie: bool, image_name: str | None = None) -> bool:
	"""在独立 globals 中 exec test.py，然后覆盖开关并调用 run_single_image_pipeline().

	这样做的好处是可以复用 test.py 中已经写好的路径解析、权重加载与可视化逻辑，
	避免 test.py 和 test-compri.py 各维护一套几乎重复的配置。
	"""
	src = TEST_PY.read_text()
	g: dict = {
		"__file__": str(TEST_PY),
		"__name__": "test_module",
	}
	try:
		exec(compile(src, str(TEST_PY), "exec"), g)
		# 覆盖开关
		if image_name is not None:
			g["TARGET_IMAGE_NAME"] = image_name
		g["ENABLE_MSICN"] = bool(enable_msicn)
		g["ENABLE_FIE"] = bool(enable_fie)

		print(f"\n=== Running: msicn={'ON' if enable_msicn else 'OFF'} | fie={'ON' if enable_fie else 'OFF'} ===\n")
		start = time.time()
		g["run_single_image_pipeline"]()
		elapsed = time.time() - start
		print(f"Finished in {elapsed:.1f}s\n")
		return True
	except Exception:
		traceback.print_exc()
		return False


def list_results():
	vis_dir = ROOT.parent / "results" / "single_image_pipeline" / "pipeline_visualization"
	if not vis_dir.exists():
		print("No visualization directory found yet:", vis_dir)
		return
	print("\nGenerated files:")
	for p in sorted(vis_dir.iterdir()):
		print(" -", p.name)


def main():
	# 四种组合对应常见消融观察：
	#   (False, False): 纯 YOLO 基线
	#   (True,  False): 只看 MSICN 的贡献
	#   (False, True ): 只看 FIE 的贡献
	#   (True,  True ): 完整 ICFIE-YOLO
	combos = [
		(False, False),
		(True, False),
		(False, True),
		(True, True),
	]

	for msicn, fie in combos:
		ok = _exec_test_with_flags(msicn, fie)
		if not ok:
			print(f"Variant msicn={msicn} fie={fie} failed — continuing to next.")

	list_results()


if __name__ == "__main__":
	main()

