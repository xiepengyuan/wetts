model_name="enqi-vits"
version="1.0"
export_path="/data1/xiepengyuan/workspace/audio/wetts/deploy/torchserve/enqi-vits_20230613"
handler="/data1/xiepengyuan/workspace/audio/wetts/deploy/torchserve/model_handler.py"
resource_dir="/data1/xiepengyuan/workspace/audio/wetts/resources/v1"

torch-model-archiver \
  --model-name $model_name \
  --version $version \
  --force \
  --export-path $export_path \
  --handler $handler \
  --extra-files $resource_dir