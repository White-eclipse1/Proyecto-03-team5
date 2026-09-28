variable "name" {
  description = "Environment-specific resource name."
  type        = string
}

variable "cidr_block" {
  description = "IPv4 /16 CIDR for the environment VPC."
  type        = string
}

variable "availability_zones" {
  description = "Two distinct availability zones in the provider region."
  type        = list(string)

  validation {
    condition     = length(var.availability_zones) == 2 && length(distinct(var.availability_zones)) == 2
    error_message = "Provide exactly two distinct availability zones."
  }
}
