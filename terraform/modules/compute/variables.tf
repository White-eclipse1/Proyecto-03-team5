variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "vpc_cidr" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "ami_id" {
  description = "Existing x86_64 AMI in the selected region; no image is built by this module."
  type        = string
}
