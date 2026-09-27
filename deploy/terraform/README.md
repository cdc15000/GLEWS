# GLEWS Terraform — AWS ECS Fargate

Deploy the Glacier Early Warning System on AWS using ECS Fargate.

## Architecture

- **ECS Fargate cluster** with two task definitions:
  - `dashboard` — long-running service behind an Application Load Balancer
  - `monitor` — scheduled via EventBridge (default: every 6 hours)
- **EFS** for persistent shared storage (`/app/data` and `/app/output`)
- **CloudWatch Logs** with 30-day retention
- **ALB** with optional HTTPS via ACM certificate

## Prerequisites

1. An AWS account with appropriate permissions.
2. A VPC with at least two public and two private subnets.
3. Earthdata credentials stored in **AWS Secrets Manager** (one secret per
   value — pass their ARNs via `earthdata_user_arn` / `earthdata_pass_arn`).
4. The GLEWS container image pushed to a registry reachable from ECS
   (ECR, GHCR, Docker Hub, etc.).
5. (Optional) An ACM certificate for TLS termination on the ALB.

## Usage

```bash
cd deploy/terraform

terraform init

# Create a .tfvars file (never commit it):
cat > prod.tfvars <<'EOF'
aws_region         = "us-east-1"
vpc_id             = "vpc-0abc123..."
private_subnet_ids = ["subnet-aaa", "subnet-bbb"]
public_subnet_ids  = ["subnet-ccc", "subnet-ddd"]
earthdata_user_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:glews/earthdata-user-XyZ"
earthdata_pass_arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:glews/earthdata-pass-AbC"
container_image    = "123456789012.dkr.ecr.us-east-1.amazonaws.com/glews:latest"
EOF

terraform plan  -var-file=prod.tfvars
terraform apply -var-file=prod.tfvars
```

## Outputs

| Output                 | Description                            |
|------------------------|----------------------------------------|
| `ecs_cluster_name`     | Name of the ECS cluster                |
| `dashboard_service_name` | ECS service running the dashboard    |
| `alb_dns_name`         | Public DNS of the load balancer        |
| `dashboard_url`        | Full URL to reach the dashboard        |
| `log_group_name`       | CloudWatch log group                   |
| `efs_file_system_id`   | EFS volume for persistent data         |

## Customization

See `variables.tf` for all tunables — task CPU/memory, replica count,
schedule expression, tags, etc.

## Teardown

```bash
terraform destroy -var-file=prod.tfvars
```

> **Note:** The EFS file system contains pipeline data. Terraform will
> destroy it. Back up `/glews-data` and `/glews-output` first if needed.
