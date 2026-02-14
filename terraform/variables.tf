variable "groq_api_key" {
  description = "The API Key for Groq AI"
  type        = string
  sensitive   = true # Hides it from logs
}