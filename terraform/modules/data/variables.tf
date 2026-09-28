variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  description = "Private subnets in two different availability zones."
  type        = list(string)
}

variable "compute_security_group_id" {
  type = string
}
