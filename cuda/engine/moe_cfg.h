/* moe_forward_sm90 GEMM config (see moe.cu): E18's microbench pick on H100 (experiments/E18/best_cfg.txt) */
#define MOE_SM90_CFG moe::CfgWS<160, 6, 160, 6, true>
