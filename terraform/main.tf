provider "aws" {
  region = "ap-south-1"  # Change this if you are not in Mumbai
}

# 1. GET THE LATEST UBUNTU AMI (AUTOMATICALLY)
# Instead of googling "ami-0f5ee...", we ask AWS for the latest ID.
data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical (Official Ubuntu Creator)
}

# 2. CREATE A SECURITY GROUP (FIREWALL)
resource "aws_security_group" "fiscalmind_sg" {
  name        = "fiscalmind-sg"
  description = "Allow SSH and Streamlit traffic"

  # SSH (Port 22)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP (Port 80)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Streamlit (Port 8501)
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound (Allow all traffic out)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 3. LAUNCH THE SERVER
resource "aws_instance" "app_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro" # Free tier eligible
  
  # KEY PAIR: Ensure this matches the key name you created in AWS Console earlier!
  key_name      = "samad-aws-key" 

  vpc_security_group_ids = [aws_security_group.fiscalmind_sg.id]

  # --- NEW PART: USER DATA ---
  user_data = templatefile("userdata.sh", {
    groq_api_key = var.groq_api_key
  })
  
  # This tells Terraform: "If I change the script, destroy and recreate the server"
  user_data_replace_on_change = true

  tags = {
    Name = "FiscalMind-Production-Auto"
  }
}

# 4. OUTPUT THE IP ADDRESS
output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.app_server.public_ip
}