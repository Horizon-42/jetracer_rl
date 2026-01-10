# 从头训练：各 reward 推荐参数（可直接运行）

> 说明
>
>- 下方命令都用 `python3`（这个环境里 `python` 可能不存在）。
>- 端口已为每个 reward 预先分配为 **不同值**（避免 서로冲突）。如果你本机某个端口已被占用，换成另一个未占用的高位端口即可。
>- 注意：`--domain-rand` 和 `--random-friction` 在本项目里是 **“禁用开关”**（默认是开启的）。
>  - 不加任何参数：domain rand / random friction **默认开启**（推荐）。
>  - 只有当你想禁用它们时才加：`--domain-rand --random-friction`。
>- `--sim-io-timeout-s` 用于防止 Unity 断连后 step/reset 卡死。

---

## base（最基础 reward）

```bash
python3 train_jetracer_centerline.py \
  --reward-type base \
  --max-cte 3.0 \
  --total-timesteps 200000 \
  --perspective-transform \
  --sim-io-timeout-s 20 \
  --port 9091 \
  
  --run-name scratch_baseline
```

## with perspective transform
python3 train_jetracer_centerline.py \
  --reward-type base \
  --max-cte 3.0 \
  --total-timesteps 200000 \
  --sim-io-timeout-s 20 \
  --port 10091 \
  --run-name scratch_base_pers

---

## track_limit（race shaping + offtrack 每步惩罚）

> 该 reward 最敏感的参数是 `--offtrack-step-penalty`：太大容易“保守停滞”，太小则容易贴边乱跑。

```bash
python3 train_jetracer_centerline.py \
  --reward-type track_limit \
  --max-cte 3.0 \
  --offtrack-step-penalty 3.0 \
  --total-timesteps 200000 \
  --sim-io-timeout-s 20 \
  --port 11091 \
  --run-name scratch_track_limit
```

---

## deepracer（DeepRacer 风格分段中心线奖励 + speed）

```bash
python3 train_jetracer_centerline.py \
  --reward-type deepracer \
  --max-cte 3.0 \
  --total-timesteps 200000 \
  --sim-io-timeout-s 20 \
  --port 12091 \
  --run-name scratch_deepracer
```

---

## centerline_v2（中心线 + speed + smooth + caution + anti-stall）

> v2 的可调参数较少，但“caution 太强”会导致慢/停。下面这组偏向：更愿意给油 + 仍保持谨慎。

```bash
python3 train_jetracer_centerline.py \
  --reward-type centerline_v2 \
  --max-cte 3.0 \
  --v2-w-speed 1.0 \
  --v2-w-caution 0.35 \
  --v2-min-speed 0.25 \
  --total-timesteps 200000 \
  --fast \
  --sim-io-timeout-s 20 \
  --port 8091 \
  --run-name scratch_centerline
```

---

## centerline_v3（更简单 + 更强 anti-stall + alive bonus）

> 如果你要 **严格 `max_cte=3`** 还不想停滞，v3 通常比 v2 更稳。

```bash
python3 train_jetracer_centerline.py \
  --reward-type centerline_v3 \
  --max-cte 3.0 \
  --v3-w-speed 1.4 \
  --v3-min-speed 0.30 \
  --v3-w-stall 4.0 \
  --v3-alive-bonus 0.05 \
  --total-timesteps 400000 \
  --sim-io-timeout-s 20 \
  --port 13051 \
  --run-name scratch_centerline_v3
```

---

## centerline_v4（中心线 + smooth + speed + alive + anti-stall，参数少）

> v4 额外包含平滑项（更抗“蛇形”），同时也包含 anti-stall 和 alive bonus。

```bash
python3 train_jetracer_centerline.py \
  --reward-type centerline_v4 \
  --max-cte 3.0 \
  --v4-w-speed 1.1 \
  --v4-w-smooth 0.20 \
  --v4-min-speed 0.28 \
  --v4-w-stall 4.0 \
  --v4-alive-bonus 0.04 \
  --total-timesteps 200000 \
  --sim-io-timeout-s 20 \
  --port 12061 \
  --run-name scratch_centerline_v4
```

---

## 端口与并行运行注意事项

- 同一时间只跑一个训练：确保只有一个 Unity DonkeySim 实例在跑，并且它监听的端口与你命令里的 `--port` 一致。
- 如果你需要同时跑 train/eval（`--eval`），会额外占用 `eval-port`（默认是 `port+1`），且需要第二个 sim。
- 如果出现 Unity 报 `Address already in use`（你之前遇到过 9078 冲突），通常是多开 sim 导致的：关掉多余的 sim 实例再训练。
