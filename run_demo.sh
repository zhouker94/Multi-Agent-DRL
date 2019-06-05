docker build -t tf-multi-agent-demo .

docker volume create log

docker run --rm \
    -v log:/log \
    tf-multi-agent-demo python main_loop.py \
        --model $1 \
        --n_agent $2 \
        --sustainable_weight $3 \
        --learn_mode $4 \
        --version $5
