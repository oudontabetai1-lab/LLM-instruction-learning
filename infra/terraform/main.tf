terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- データセット・チェックポイント・評価レポートの保管先 ---
resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${var.project_name}-artifacts-"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- パイプラインのコンテナイメージ ---
resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.project_name}-pipeline"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# --- GPU インスタンス(Ollama サーバー + 学習ジョブ) ---
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
}

resource "aws_security_group" "gpu" {
  name_prefix = "${var.project_name}-gpu-"
  description = "GPU instance for Ollama and training"

  # SSH は許可された CIDR のみ。Ollama ポートは外部公開しない(SSM/トンネル経由)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_iam_role" "gpu" {
  name_prefix = "${var.project_name}-gpu-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "gpu_s3" {
  name_prefix = "s3-artifacts-"
  role        = aws_iam_role.gpu.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      }
    ]
  })
}

# Bedrock を教師モデルに使う場合の推論権限(var.enable_bedrock_teacher で有効化)
resource "aws_iam_role_policy" "gpu_bedrock" {
  count       = var.enable_bedrock_teacher ? 1 : 0
  name_prefix = "bedrock-invoke-"
  role        = aws_iam_role.gpu.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse"]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "gpu_ssm" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu" {
  name_prefix = "${var.project_name}-gpu-"
  role        = aws_iam_role.gpu.name
}

resource "aws_instance" "gpu" {
  count                  = var.gpu_instance_count
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.gpu_instance_type
  iam_instance_profile   = aws_iam_instance_profile.gpu.name
  vpc_security_group_ids = [aws_security_group.gpu.id]
  key_name               = var.key_name != "" ? var.key_name : null

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -eux
    curl -fsSL https://ollama.com/install.sh | sh
    systemctl enable --now ollama
  EOF

  tags = {
    Name    = "${var.project_name}-gpu-${count.index}"
    Project = var.project_name
  }
}
