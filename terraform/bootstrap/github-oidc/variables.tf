variable "existing_oidc_provider_arn" {
  description = "Optional ARN of an existing GitHub OIDC provider in the target account. Reused without management or import; verify its URL and audience before use."
  type        = string
  default     = null

  validation {
    condition = var.existing_oidc_provider_arn == null ? true : can(regex(
      "^arn:[^:]+:iam::[0-9]{12}:oidc-provider/token\\.actions\\.githubusercontent\\.com$",
      var.existing_oidc_provider_arn
    ))
    error_message = "Provide a GitHub OIDC provider ARN or leave null to define a new provider."
  }
}
