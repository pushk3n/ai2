# Project Guidelines

## Code Style
- 代码注释和文档说明使用中文，遵循 [docs/prd2.md](docs/prd2.md)。
- 配置优先使用显式 dataclass 注入，不要把关键配置隐藏在 kwargs 或隐式默认值里；参考 [src/icfie_yolo.py](src/icfie_yolo.py) 和 [src/msicn.py](src/msicn.py)。
- 修改推理或可视化代码时，保留数值范围、通道对齐和形状断言这类安全检查，不要为了“先跑通”删除它们。

## Architecture
- [src](src) 是自研 ICFIE-YOLO 适配与实验代码，核心链路是 MSICN -> YOLO backbone/neck -> FIE -> detect head；入口与适配模式见 [src/icfie_yolo.py](src/icfie_yolo.py) 和 [docs/prd2.md](docs/prd2.md)。
- [yolov7](yolov7) 视为外部依赖和参考实现。除非任务明确要求修改 YOLOv7 本体，否则优先在 [src](src) 层做适配，不要把项目约定散落到上游代码里。
- 处理可视化或检测结果时，最终展示应基于 YOLOv7 官方推理链路的 letterbox、NMS 和回投原图坐标；原始检测头响应只作为调试信号，相关背景见 [src/yolo_test.py](src/yolo_test.py)。
- Matplotlib 中文标题需要显式配置 CJK 字体；相关处理已在 [src/infer.py](src/infer.py) 中实现，修改图像输出逻辑时不要移除。

## Build and Test
- 环境初始化和依赖安装以 [README.md](README.md) 为准：创建 Python 3.10 环境、安装 [requirements.txt](requirements.txt) 和 [yolov7/requirements.txt](yolov7/requirements.txt)，并将 yolov7.pt 放入 [yolov7](yolov7)。
- 常用验证命令在 [README.md](README.md) 中已有说明。优先使用 [src/mock.py](src/mock.py) 做轻量冒烟验证，再根据任务使用 [src/infer.py](src/infer.py)、[src/test.py](src/test.py) 或 [src/yolo_test.py](src/yolo_test.py)。
- 这类脚本通常从 [src](src) 目录运行；改动 import、路径或资源加载逻辑时，先检查运行目录假设是否被破坏。

## Conventions
- 项目重点是“显式解耦”和“显式依赖注入”：MSICN、backbone/neck、FIE、detect head 应保持边界清晰，避免跨模块偷传状态；约束来源见 [docs/prd2.md](docs/prd2.md)。
- 训练阶段控制依赖明确的 requires_grad 和 no_grad 语义，不能只靠 optimizer 参数列表模拟冻结；如果实现训练脚本，按 [docs/prd2.md](docs/prd2.md) 的三阶段要求落地。
- MSICN 输出必须保持在 [0, 1]，FIE 输出通道必须与 detect head 输入对齐。涉及这两个模块时，优先修根因，不要靠临时 reshape、broadcast 或静默截断掩盖问题。

## Documentation
- 环境准备和当前进度看 [README.md](README.md)。
- 架构与训练规范以 [docs/prd2.md](docs/prd2.md) 为主。
- 论文背景与方法说明看 [docs/md.md](docs/md.md), [docs/deepseek.html](docs/deepseek.html), [docs/gpt.html](docs/gpt.html)。