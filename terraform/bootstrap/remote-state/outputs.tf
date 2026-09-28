output "state_bucket_name" {
  description = "Supply this generated name to dev/prod init using -backend-config=bucket=..."
  value       = aws_s3_bucket.state.id
}

output "state_bucket_region" {
  description = "Supply this region to dev/prod init using -backend-config=region=..."
  value       = var.region
}
