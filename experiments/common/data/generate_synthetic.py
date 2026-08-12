"""生成正式合成数据并固定数据划分。先完成本文件，再做 E1-E3。

每个样本至少保存：
- clean_true
- background_true
- observed = clean_true + background_true + noise
- 背景源中心、尺度、方向
- 星点参数、基础星图 ID、随机种子和 split

TODO:
1. 明确 clean_true 的来源与使用许可。
2. 实现单中心径向、偏心/椭圆、多光源或非均匀平滑背景。
3. 加入可控的污染强度、尺度和噪声水平。
4. 先按基础星图划分，再生成污染版本，防止数据泄漏。
5. 生成固定 manifest，后续实验只读取，不临时重新划分。
"""

from __future__ import annotations


def generate_sample(*, seed: int, split: str) -> dict:
    """生成一个样本及其完整真值和元数据。"""
    # TODO: 返回数组或保存路径，不要只返回展示图。
    raise NotImplementedError


def main() -> None:
    # TODO: 固定 60/20/20 或最终确认的划分，并写出 manifest.json/csv。
    raise NotImplementedError


if __name__ == "__main__":
    main()
