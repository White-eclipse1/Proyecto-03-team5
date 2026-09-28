variable "region" {
  description = "AWS region. Availability zones and AMI must belong to this region."
  type        = string
  default     = "us-east-1"
}

variable "availability_zones" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b"]
}

variable "ami_id" {
  description = "Existing x86_64 AMI for EC2. Not required to run init or validate."
  type        = string
}
