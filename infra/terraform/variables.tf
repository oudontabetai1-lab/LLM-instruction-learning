variable "project_name" {
  description = "リソース名のプレフィックス"
  type        = string
  default     = "llm-itl"
}

variable "aws_region" {
  description = "デプロイ先リージョン"
  type        = string
  default     = "ap-northeast-1"
}

variable "gpu_instance_type" {
  description = "GPU インスタンスタイプ。70B 級の教師を動かすなら g5.12xlarge / g6e.12xlarge 以上を推奨"
  type        = string
  default     = "g5.2xlarge"
}

variable "gpu_instance_count" {
  description = "GPU インスタンス数(0 でインスタンスなし=ストレージのみ作成)"
  type        = number
  default     = 1
}

variable "root_volume_gb" {
  description = "ルートボリュームサイズ(モデル保管用に大きめに)"
  type        = number
  default     = 300
}

variable "key_name" {
  description = "SSH キーペア名(空なら SSM Session Manager のみで接続)"
  type        = string
  default     = ""
}

variable "ssh_allowed_cidrs" {
  description = "SSH を許可する CIDR(既定では全拒否。SSM 接続を推奨)"
  type        = list(string)
  default     = []
}

variable "enable_bedrock_teacher" {
  description = "教師モデルに Bedrock を使う場合 true(インスタンスに Bedrock 推論権限を付与)"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "GPU セキュリティグループを作成する VPC ID(空ならデフォルト VPC を使用)"
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "GPU インスタンスを配置するサブネット ID(空ならデフォルト VPC のデフォルトサブネットを使用)"
  type        = string
  default     = ""
}
