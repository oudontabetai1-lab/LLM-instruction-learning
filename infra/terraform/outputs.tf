output "artifacts_bucket" {
  description = "データセット・チェックポイント保管用 S3 バケット名"
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  description = "パイプラインイメージの ECR リポジトリ URL"
  value       = aws_ecr_repository.pipeline.repository_url
}

output "gpu_instance_ids" {
  description = "GPU インスタンス ID(SSM 接続: aws ssm start-session --target <id>)"
  value       = aws_instance.gpu[*].id
}
