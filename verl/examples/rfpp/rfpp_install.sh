pip install torch=2.6.0

pip install vllm

cd /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl

pip install -e .

pip install /lpai/volumes/base-rlhf-ali-sh/yanglele/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install tensordict==0.6.2

cp /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/backup/modeling_llava.py $CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py
cp /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/backup/configuration_llava.py $CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py



cd /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/examples/rfpp

bash /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/examples/rfpp/run_dsv.sh > run_dsv.log 2>&1 &




sleep 100000
