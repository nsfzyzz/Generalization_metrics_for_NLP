import os

#CKPT_DIR = "/work/rkunani/pytorch-transformer/checkpoint_yaoqing"
CKPT_DIR = "/home/ubuntu/rkunani/checkpoint"

"""
EXPERIMENTS = {
    "IWSLT": [
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth6_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth6_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth6_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth6_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth6_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth7_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth7_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth7_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth7_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth7_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth8_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth8_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth8_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth8_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample40000_depth8_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth4_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth4_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth4_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth4_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth4_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth5_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth5_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth5_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth5_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth5_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth6_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth6_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth6_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth6_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth6_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth7_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth7_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth7_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth7_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth7_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth4_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth4_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth4_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth4_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth4_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth5_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth5_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth5_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth5_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth5_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth6_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth6_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth6_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth6_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth6_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth7_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth7_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth7_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth7_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth7_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth8_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth8_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth8_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth8_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample120000_depth8_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth4_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth4_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth4_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth4_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth4_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth5_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth5_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth5_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth5_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth5_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth6_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth6_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth6_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth6_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth6_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth7_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth7_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth7_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth7_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth7_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth8_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth8_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth8_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth8_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample160000_depth8_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth5_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth5_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth5_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth5_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth5_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth6_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth6_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth6_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth6_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth6_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth7_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth7_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth7_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth7_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth7_lr2.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth8_lr0.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth8_lr0.75_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth8_lr1.0_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth8_lr1.5_dropout0.1"),
        os.path.join(CKPT_DIR, "IWSLT_sample200000_depth8_lr2.0_dropout0.1"),
    ],
"""
    
CKPT_DIR = "/work/yyaoqing/Good_vs_bad_data/checkpoint/NMT_epochs/Simpson"

EXPERIMENTS = { "WMT": [
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth5_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth6_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth7_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample1280000_depth8_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth5_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth6_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth7_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample160000_depth8_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth5_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth6_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth7_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample2560000_depth8_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth5_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth6_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth7_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample320000_depth8_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth5_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth6_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth7_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.0625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.125_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.25_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.375_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.625_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "WMT14_sample640000_depth8_lr2.0_dropout0.1"),
]}