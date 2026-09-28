output "vpc_id" {
  value = aws_vpc.this.id
}

output "cidr_block" {
  value = aws_vpc.this.cidr_block
}

output "subnet_ids" {
  value = aws_subnet.this[*].id
}

output "s3_endpoint_id" {
  value = aws_vpc_endpoint.s3.id
}
