terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# Independent bucket: never adopts or references the existing P2-04 buckets.
resource "aws_s3_bucket" "this" {
  bucket_prefix = "${var.name}-artifacts-"
  force_destroy = false

  tags = { Name = "${var.name}-artifacts" }
}

resource "aws_s3_bucket_public_access_block" "this" {
  bucket = aws_s3_bucket.this.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "this" {
  bucket = aws_s3_bucket.this.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.this.arn, "${aws_s3_bucket.this.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

# Dedicated destination in the same account/region. Do not log this bucket
# back to itself or to the source: that would generate recursive access logs.
resource "aws_s3_bucket" "logs" {
  bucket_prefix = "${var.name}-access-logs-"
  force_destroy = false

  tags = { Name = "${var.name}-access-logs" }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Resolved by AWS only during a future execution; no account ID is hardcoded.
data "aws_caller_identity" "current" {}

resource "aws_s3_bucket_policy" "logs" {
  bucket = aws_s3_bucket.logs.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AllowS3AccessLogDelivery"
        Effect    = "Allow"
        Principal = { Service = "logging.s3.amazonaws.com" }
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.logs.arn}/access-logs/*"
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = [
              aws_s3_bucket.this.arn,
              aws_s3_bucket.dvc_cache.arn,
              aws_s3_bucket.dataset_releases.arn,
            ]
          }
          StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        }
      },
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.logs.arn, "${aws_s3_bucket.logs.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      }
    ]
  })
}

resource "aws_s3_bucket_logging" "this" {
  bucket        = aws_s3_bucket.this.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "access-logs/"

  depends_on = [
    aws_s3_bucket_policy.logs,
    aws_s3_bucket_public_access_block.logs,
    aws_s3_bucket_server_side_encryption_configuration.logs,
  ]
}

# P2-15: dataset buckets with versioning enabled, so an overwritten or
# deleted object can be recovered. Independent resources from "this" above
# (P2-06's artifacts bucket) — same security baseline (encryption, no
# public access, HTTPS-only, access logs delivered to the same "logs"
# bucket as "this", under their own prefix).

resource "aws_s3_bucket" "dvc_cache" {
  bucket_prefix = "${var.name}-dvc-cache-"
  force_destroy = false

  tags = { Name = "${var.name}-dvc-cache" }
}

resource "aws_s3_bucket_versioning" "dvc_cache" {
  bucket = aws_s3_bucket.dvc_cache.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "dvc_cache" {
  bucket = aws_s3_bucket.dvc_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc_cache" {
  bucket = aws_s3_bucket.dvc_cache.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "dvc_cache" {
  bucket = aws_s3_bucket.dvc_cache.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.dvc_cache.arn, "${aws_s3_bucket.dvc_cache.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_logging" "dvc_cache" {
  bucket        = aws_s3_bucket.dvc_cache.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "access-logs/dvc-cache/"

  depends_on = [
    aws_s3_bucket_policy.logs,
    aws_s3_bucket_public_access_block.logs,
    aws_s3_bucket_server_side_encryption_configuration.logs,
  ]
}

resource "aws_s3_bucket" "dataset_releases" {
  bucket_prefix = "${var.name}-dataset-releases-"
  force_destroy = false

  tags = { Name = "${var.name}-dataset-releases" }
}

resource "aws_s3_bucket_versioning" "dataset_releases" {
  bucket = aws_s3_bucket.dataset_releases.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "dataset_releases" {
  bucket = aws_s3_bucket.dataset_releases.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dataset_releases" {
  bucket = aws_s3_bucket.dataset_releases.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "dataset_releases" {
  bucket = aws_s3_bucket.dataset_releases.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.dataset_releases.arn, "${aws_s3_bucket.dataset_releases.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_s3_bucket_logging" "dataset_releases" {
  bucket        = aws_s3_bucket.dataset_releases.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "access-logs/dataset-releases/"

  depends_on = [
    aws_s3_bucket_policy.logs,
    aws_s3_bucket_public_access_block.logs,
    aws_s3_bucket_server_side_encryption_configuration.logs,
  ]
}
