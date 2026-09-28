output "bucket_name" {
  value = aws_s3_bucket.this.id
}

output "dvc_bucket_names" {
  value = {
    dvc-cache        = aws_s3_bucket.dvc_cache.id
    dataset-releases = aws_s3_bucket.dataset_releases.id
  }
}
