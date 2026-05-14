"""test-compri.py

一次性对比运行 src/test.py 中的四种开关组合并保存输出：
 1) 纯 YOLO
 2) MSICN 开，FIE 关
 3) MSICN 关，FIE 开
 4) 全开

脚本会逐个加载 src/test.py 的代码（以独立 globals 执行），在每次调用前设置
`ENABLE_MSICN` / `ENABLE_FIE` 两个全局变量，从而复用 `run_single_image_pipeline()`。
运行完成后，会列出 results/single_image_pipeline/pipeline_visualization 下新增的文件。
"""

from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
TEST_PY = ROOT / "test.py"


def _exec_test_with_flags(enable_msicn: bool, enable_fie: bool, image_name: str | None = None) -> bool:
	"""在独立 globals 中 exec test.py，然后设置 flags 并调用 run_single_image_pipeline()."""
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

