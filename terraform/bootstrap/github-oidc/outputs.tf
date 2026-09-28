output "role_arn" {
  description = "After an authorized future bootstrap, set this value as the GitHub repository variable AWS_ROLE_ARN."
  value       = aws_iam_role.github_actions.arn
}
