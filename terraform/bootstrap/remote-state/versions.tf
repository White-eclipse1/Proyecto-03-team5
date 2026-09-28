terraform {
  required_version = ">= 1.10, < 2.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# Local backend intentionally: this root creates the remote backend's bucket.
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project = "mlops-p2"
      Purpose = "terraform-state"
    }
  }
}
