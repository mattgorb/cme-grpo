terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
  }
}

############################
# Variables
############################
variable "aws_region"    { default = "us-east-1" }
variable "instance_type" { default = "g6e.xlarge" }
variable "name_prefix"   { default = "cme-grpo" }

provider "aws" {
  region = var.aws_region
}

resource "random_pet" "suffix" {
  length = 2
}

############################
# Networking: VPC + public subnet + IGW + route
############################
data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "vpc" {
  cidr_block           = "10.77.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = { Name = "${var.name_prefix}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.vpc.id
  tags   = { Name = "${var.name_prefix}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.vpc.id
  cidr_block              = "10.77.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available.names[0]
  tags = { Name = "${var.name_prefix}-subnet" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.name_prefix}-rt" }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

############################
# Security Group: SSH + all egress
############################
resource "aws_security_group" "sg" {
  name        = "${var.name_prefix}-sg"
  description = "Allow SSH and outbound internet"
  vpc_id      = aws_vpc.vpc.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-sg" }
}

############################
# IAM for SSM Session Manager
############################
data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ec2_role" {
  name               = "${var.name_prefix}-role-${random_pet.suffix.id}"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "profile" {
  name = "${var.name_prefix}-profile-${random_pet.suffix.id}"
  role = aws_iam_role.ec2_role.name
}

############################
# Use existing SSH key pair
############################
data "aws_key_pair" "existing" {
  key_name = "id_rsa"
}

############################
# Deep Learning AMI (GPU / PyTorch) - Amazon Linux 2023
# Supports: G4dn, G5, G6, Gr6, G6e, P4, P4de, P5, P5e, P5en, P6-B200
############################
resource "aws_instance" "gpu" {
  ami                         = "ami-0025c2ddec381a62b"
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.sg.id]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.profile.name
  key_name                    = data.aws_key_pair.existing.key_name

  root_block_device {
    volume_size = 400
    volume_type = "gp3"
  }

  tags = { Name = "${var.name_prefix}-gpu-${random_pet.suffix.id}" }
}

############################
# Outputs
############################
output "instance_id" {
  value = aws_instance.gpu.id
}

output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/id_rsa ec2-user@${aws_instance.gpu.public_ip}"
}

output "how_to_connect_via_ssm" {
  value = "aws ssm start-session --target ${aws_instance.gpu.id}"
}
