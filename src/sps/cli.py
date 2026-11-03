"""Command-line interface for the smart photo selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .image_ops import list_images
from .pipeline import SelectionConfig, SmartPhotoSelector, save_results
from .ranking import PairwiseRanker
from .scoring import ScoringWeights


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sps",
        description="智能选片：从同一场景的一组照片中选出最佳照片与取景框",
    )
    parser.add_argument("--version", action="version", version=f"sps {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="生成合成演示序列并运行选片")
    demo.add_argument("--out", default="outputs/demo", help="输出目录")
    demo.add_argument("--series-dir", default="examples/demo_series",
                      help="演示图片输出目录")

    select = sub.add_parser("select", help="对目录中的一组照片进行选片")
    select.add_argument("--input", required=True, help="同一场景的照片目录")
    select.add_argument("--out", default="outputs/run", help="输出目录")
    select.add_argument("--ranker", default=None, help="训练好的 PairwiseRanker JSON")
    select.add_argument("--context", default="", help="场景描述（用于 VLM 评审）")
    select.add_argument("--weights", default="0.30,0.20,0.30,0.20",
                        help="technical,aesthetic,composition,semantic 权重")
    select.add_argument("--max-candidates", type=int, default=80,
                        help="每张照片最多保留的候选框数量")
    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as demo_main
    demo_main(out_dir=args.out, series_dir=args.series_dir)
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    paths = list_images(input_dir)
    if not paths:
        print(f"未在 {input_dir} 中找到图片", file=sys.stderr)
        return 2

    weights = ScoringWeights(*[float(v) for v in args.weights.split(",")])
    config = SelectionConfig(weights=weights, context=args.context)
    config.candidate.max_candidates = args.max_candidates
    ranker = PairwiseRanker.load(args.ranker) if args.ranker else None

    selector = SmartPhotoSelector(config=config, ranker=ranker)
    series = selector.select_series(paths)
    outputs = save_results(series, args.out)

    print(f"最佳照片：{series.selected.path.name}")
    print(f"最佳取景框：{series.selected.best.crop.box}")
    for photo in series.ranking:
        print(f"  #{photo.rank} {photo.path.name:<32} "
              f"total={photo.best.score.total:.4f} series={photo.series_score:.4f}")
    print(json.dumps({k: str(v) for k, v in outputs.items()},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "select":
        return _cmd_select(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
