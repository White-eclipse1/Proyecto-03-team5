terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.name }
}

# RDS requires subnets in two availability zones. No public routes are added.
resource "aws_subnet" "this" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.cidr_block, 8, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = false

  tags = { Name = "${var.name}-${count.index + 1}" }
}

# Resolved by AWS only during a future execution; no region is hardcoded.
data "aws_region" "current" {}

# P2-15: Gateway endpoint so traffic to S3 (the DVC buckets in modules/storage)
# never leaves through a NAT/Internet gateway. Neither exists in this module
# (see comment above: no public routes), so this only affects the VPC's
# default route table, which is what the subnets use implicitly.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${data.aws_region.current.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_vpc.this.default_route_table_id]

  tags = { Name = "${var.name}-s3-gateway-endpoint" }
}
