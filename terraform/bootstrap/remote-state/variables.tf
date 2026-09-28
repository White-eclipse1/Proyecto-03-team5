variable "region" {
  description = "AWS region for the dedicated Terraform state bucket."
  type        = string
  default     = "us-east-1"
}

variable "bucket_prefix" {
  description = "Dedicated state bucket prefix; AWS provider appends a unique suffix."
  type        = string
  default     = "mlops-p2-tfstate-"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,36}$", var.bucket_prefix))
    error_message = "Use 1-37 lowercase letters, digits or hyphens, starting with a letter or digit."
  }
}
