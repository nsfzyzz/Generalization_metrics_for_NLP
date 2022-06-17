import os

CKPT_DIR = "/work/rkunani/pytorch-transformer/checkpoint"

EXPERIMENTS = [
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
    os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample200000_depth4_lr1.5_dropout0.1"),
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
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth4_lr2.0_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample40000_depth5_lr0.5_dropout0.1"),
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
    os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr0.75_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr0.5_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr1.0_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr1.5_dropout0.1"),
    os.path.join(CKPT_DIR, "IWSLT_sample80000_depth8_lr2.0_dropout0.1"),
]