#!/bin/bash

# 批量启动多个训练任务的脚本
# 用法: 
#   ./launch_multiple_trainings.sh [后缀]                    # 启动所有训练
#   ./launch_multiple_trainings.sh [后缀] --interactive      # 交互式选择要启动的训练
#
# 例如: ./launch_multiple_trainings.sh exp001
# 如果未提供后缀，则使用当前时间戳

set -e

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 解析参数
INTERACTIVE=false
SUFFIX=""

if [ "$1" == "--interactive" ]; then
    INTERACTIVE=true
    SUFFIX="$2"
elif [ "$2" == "--interactive" ]; then
    INTERACTIVE=true
    SUFFIX="$1"
else
    SUFFIX="$1"
fi

# 获取统一的后缀
if [ -z "$SUFFIX" ]; then
    SUFFIX="$(date +%Y%m%d_%H%M%S)"
    echo "未提供后缀，使用时间戳: $SUFFIX"
else
    echo "使用提供的后缀: $SUFFIX"
fi

# 检查是否使用 gnome-terminal 还是其他终端模拟器
USE_GNOME_TERMINAL=false
USE_XTERM=false

if command -v gnome-terminal &> /dev/null; then
    USE_GNOME_TERMINAL=true
    echo "检测到 gnome-terminal，将在新终端窗口中运行训练"
elif command -v xterm &> /dev/null; then
    USE_XTERM=true
    echo "检测到 xterm，将在新终端窗口中运行训练"
else
    echo "未检测到图形终端，将使用后台进程运行（输出将保存到日志文件）"
fi

# 训练命令数组（带描述）
declare -a TRAIN_DESCRIPTIONS=(
    "base (最基础 reward, with perspective-transform)"
    "base with perspective transform (文档中的第二个变体)"
    "track_limit (race shaping + offtrack 每步惩罚)"
    "deepracer (DeepRacer 风格分段中心线奖励 + speed)"
    "centerline_v2 (中心线 + speed + smooth + caution + anti-stall)"
    "centerline_v3 (更简单 + 更强 anti-stall + alive bonus)"
    "centerline_v4 (中心线 + smooth + speed + alive + anti-stall)"
)

declare -a TRAIN_COMMANDS=(
    # base (最基础 reward)
    # "python3 train_jetracer_centerline.py --reward-type base --max-cte 3.0 --total-timesteps 200000 --perspective-transform --sim-io-timeout-s 20 --port 9091 --run-name scratch_baseline_${SUFFIX}"
    
    # base with perspective transform (第二个base变体，注意文档中这个命令缺少 --perspective-transform，但标题说有)
    "python3 train_jetracer_centerline.py --reward-type base --max-cte 3.0 --total-timesteps 200000 --sim-io-timeout-s 20 --port 10091 --run-name scratch_base_pers_${SUFFIX}"
    
    # track_limit
    "python3 train_jetracer_centerline.py --reward-type track_limit --max-cte 3.0 --offtrack-step-penalty 3.0 --total-timesteps 200000 --sim-io-timeout-s 20 --port 11091 --run-name scratch_track_limit_${SUFFIX}"
    
    # deepracer
    "python3 train_jetracer_centerline.py --reward-type deepracer --max-cte 3.0 --total-timesteps 200000 --sim-io-timeout-s 20 --port 12091 --run-name scratch_deepracer_${SUFFIX}"
    
    # centerline_v2
    "python3 train_jetracer_centerline.py --reward-type centerline_v2 --max-cte 3.0 --v2-w-speed 1.0 --v2-w-caution 0.35 --v2-min-speed 0.25 --total-timesteps 200000 --fast --sim-io-timeout-s 20 --port 8091 --run-name scratch_centerline_${SUFFIX}"
    
    # centerline_v3 (文档中 total-timesteps 是 400000)
    "python3 train_jetracer_centerline.py --reward-type centerline_v3 --max-cte 3.0 --v3-w-speed 1.4 --v3-min-speed 0.30 --v3-w-stall 4.0 --v3-alive-bonus 0.05 --total-timesteps 400000 --sim-io-timeout-s 20 --port 13051 --run-name scratch_centerline_v3_${SUFFIX}"
    
    # centerline_v4
    "python3 train_jetracer_centerline.py --reward-type centerline_v4 --max-cte 3.0 --v4-w-speed 1.1 --v4-w-smooth 0.20 --v4-min-speed 0.28 --v4-w-stall 4.0 --v4-alive-bonus 0.04 --total-timesteps 200000 --sim-io-timeout-s 20 --port 12061 --run-name scratch_centerline_v4_${SUFFIX}"
)

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/training_logs_${SUFFIX}"
mkdir -p "$LOG_DIR"
echo "训练日志将保存到: $LOG_DIR"

# 提取 run-name 的辅助函数（兼容不同系统）
extract_run_name() {
    local cmd="$1"
    # 使用 awk 提取 --run-name 后面的值（更可靠）
    # awk 会按空格分割，找到 --run-name 参数后，返回下一个字段
    local result=$(echo "$cmd" | awk '{
        for(i=1; i<=NF; i++) {
            if($i == "--run-name" && i < NF) {
                print $(i+1)
                exit
            }
        }
    }')
    
    # 如果 awk 失败，尝试使用 sed（作为备选）
    if [ -z "$result" ]; then
        result=$(echo "$cmd" | sed -n 's/.*--run-name[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p')
    fi
    
    # 去掉可能的引号或特殊字符（去除单引号和双引号）
    result=$(echo "$result" | tr -d "'" | tr -d '"')
    
    echo "$result"
}

# 启动训练的辅助函数
launch_training() {
    local cmd="$1"
    local index="$2"
    local run_name=$(extract_run_name "$cmd")
    
    # 验证 run_name 是否成功提取
    if [ -z "$run_name" ]; then
        echo "错误: 无法从命令中提取 run-name"
        echo "命令: $cmd"
        return 1
    fi
    
    if [ "$USE_GNOME_TERMINAL" = true ]; then
        # 使用 gnome-terminal (使用 eval 来正确处理命令中的参数)
        gnome-terminal --title="Training: $run_name" -- bash -c "cd '$SCRIPT_DIR' && eval '$cmd'; echo ''; echo '训练完成！按任意键关闭窗口...'; read -n 1" &
    elif [ "$USE_XTERM" = true ]; then
        # 使用 xterm
        xterm -T "Training: $run_name" -e bash -c "cd '$SCRIPT_DIR' && eval '$cmd'; echo ''; echo '训练完成！按任意键关闭窗口...'; read -n 1" &
    else
        # 使用后台进程，输出重定向到日志文件
        local log_file="$LOG_DIR/training_${run_name}.log"
        echo "启动训练: $run_name (日志: $log_file)"
        cd "$SCRIPT_DIR" && nohup bash -c "$cmd" > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "$LOG_DIR/training_${run_name}.pid"
        echo "  进程 PID: $pid"
    fi
}

# 交互式选择要启动的训练
if [ "$INTERACTIVE" = true ]; then
    echo ""
    echo "=========================================="
    echo "请选择要启动的训练任务 (多个选择用空格分隔，例如: 1 3 5，或输入 'all' 启动全部):"
    echo "=========================================="
    for i in "${!TRAIN_DESCRIPTIONS[@]}"; do
        echo "  [$((i+1))] ${TRAIN_DESCRIPTIONS[$i]}"
    done
    echo ""
    read -p "请输入选择: " selection
    
    if [ "$selection" = "all" ] || [ "$selection" = "ALL" ]; then
        # 生成所有索引 (0 到 length-1)
        SELECTED_INDICES=()
        for ((i=0; i<${#TRAIN_COMMANDS[@]}; i++)); do
            SELECTED_INDICES+=($i)
        done
    else
        SELECTED_INDICES=($selection)
        # 将用户输入转换为0-based索引并验证
        declare -a VALID_INDICES=()
        for idx in "${SELECTED_INDICES[@]}"; do
            # 转换为0-based
            zero_based=$((idx - 1))
            if [ $zero_based -ge 0 ] && [ $zero_based -lt ${#TRAIN_COMMANDS[@]} ]; then
                VALID_INDICES+=($zero_based)
            else
                echo "警告: 无效的选择 $idx，已忽略"
            fi
        done
        SELECTED_INDICES=("${VALID_INDICES[@]}")
    fi
    
    # 创建过滤后的命令数组
    declare -a FILTERED_COMMANDS=()
    declare -a FILTERED_DESCRIPTIONS=()
    for idx in "${SELECTED_INDICES[@]}"; do
        FILTERED_COMMANDS+=("${TRAIN_COMMANDS[$idx]}")
        FILTERED_DESCRIPTIONS+=("${TRAIN_DESCRIPTIONS[$idx]}")
    done
    TRAIN_COMMANDS=("${FILTERED_COMMANDS[@]}")
    TRAIN_DESCRIPTIONS=("${FILTERED_DESCRIPTIONS[@]}")
    
    if [ ${#TRAIN_COMMANDS[@]} -eq 0 ]; then
        echo "错误: 没有选择任何训练任务"
        exit 1
    fi
    
    echo ""
    echo "已选择 ${#TRAIN_COMMANDS[@]} 个训练任务"
    echo ""
fi

# 启动所有训练
echo ""
echo "=========================================="
echo "开始启动所有训练任务..."
echo "=========================================="
echo ""

for i in "${!TRAIN_COMMANDS[@]}"; do
    cmd="${TRAIN_COMMANDS[$i]}"
    desc="${TRAIN_DESCRIPTIONS[$i]}"
    echo "[$((i+1))/${#TRAIN_COMMANDS[@]}] 启动训练..."
    echo "  描述: $desc"
    
    # 提取 run-name 用于显示
    run_name=$(extract_run_name "$cmd")
    echo "  运行名称: $run_name"
    
    launch_training "$cmd" "$i"
    
    # 稍微延迟一下，避免同时启动太多进程
    sleep 2
done

echo ""
echo "=========================================="
echo "所有训练任务已启动！"
echo "=========================================="

if [ "$USE_GNOME_TERMINAL" = false ] && [ "$USE_XTERM" = false ]; then
    echo ""
    echo "训练在后台运行，日志文件位于: $LOG_DIR"
    echo "可以使用以下命令查看进程:"
    echo "  ps aux | grep train_jetracer_centerline"
    echo ""
    echo "要停止所有训练，可以运行:"
    echo "  pkill -f train_jetracer_centerline"
    echo "或者停止特定的训练:"
    for i in "${!TRAIN_COMMANDS[@]}"; do
        run_name=$(extract_run_name "${TRAIN_COMMANDS[$i]}")
        pid_file="$LOG_DIR/training_${run_name}.pid"
        if [ -f "$pid_file" ]; then
            echo "  kill \$(cat $pid_file)  # 停止 $run_name"
        fi
    done
fi

# 等待一下，确保所有训练都已启动
sleep 3

# 启动 TensorBoard
echo ""
echo "=========================================="
echo "启动 TensorBoard..."
echo "=========================================="
echo ""

TB_LOG_DIR="$SCRIPT_DIR/tensorboard_logs"
if [ ! -d "$TB_LOG_DIR" ]; then
    echo "警告: TensorBoard 日志目录不存在: $TB_LOG_DIR"
    echo "TensorBoard 将在目录创建后显示日志"
fi

# 检查 tensorboard 是否已安装
if ! command -v tensorboard &> /dev/null; then
    echo "错误: 未找到 tensorboard 命令"
    echo "请先安装: pip install tensorboard"
    exit 1
fi

# 查找可用的端口
TB_PORT=6006
while lsof -Pi :$TB_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; do
    echo "端口 $TB_PORT 已被占用，尝试下一个端口..."
    TB_PORT=$((TB_PORT + 1))
done

echo "TensorBoard 将运行在: http://localhost:$TB_PORT"
echo "日志目录: $TB_LOG_DIR"
echo ""

if [ "$USE_GNOME_TERMINAL" = true ]; then
    gnome-terminal --title="TensorBoard - Port $TB_PORT" -- bash -c "tensorboard --logdir=$TB_LOG_DIR --port=$TB_PORT --bind_all; echo ''; echo 'TensorBoard 已停止。按任意键关闭窗口...'; read -n 1"
elif [ "$USE_XTERM" = true ]; then
    xterm -T "TensorBoard - Port $TB_PORT" -e bash -c "tensorboard --logdir=$TB_LOG_DIR --port=$TB_PORT --bind_all; echo ''; echo 'TensorBoard 已停止。按任意键关闭窗口...'; read -n 1" &
else
    # 后台运行 TensorBoard
    TB_LOG_FILE="$LOG_DIR/tensorboard.log"
    nohup tensorboard --logdir="$TB_LOG_DIR" --port="$TB_PORT" --bind_all > "$TB_LOG_FILE" 2>&1 &
    TB_PID=$!
    echo $TB_PID > "$LOG_DIR/tensorboard.pid"
    echo "TensorBoard 已在后台启动 (PID: $TB_PID)"
    echo "日志文件: $TB_LOG_FILE"
    echo ""
    echo "要停止 TensorBoard，运行:"
    echo "  kill $TB_PID"
    echo "或: kill \$(cat $LOG_DIR/tensorboard.pid)"
fi

echo ""
echo "=========================================="
echo "所有任务已启动完成！"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 访问 TensorBoard: http://localhost:$TB_PORT"
echo "  - 检查训练状态: 查看各自的终端窗口或日志文件"
echo "  - 项目目录: $SCRIPT_DIR"
echo ""

