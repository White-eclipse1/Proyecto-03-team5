terraform {
  # Supply bucket and region with terraform init -backend-config.
  backend "s3" {
    key          = "environments/prod/terraform.tfstate"
    use_lockfile = true
    encrypt      = true
  }
}
