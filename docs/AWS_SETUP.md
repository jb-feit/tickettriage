# Manual AWS setup (week 1)

No Terraform or Ansible here by design: you click through the console once, write down what
you made, and tear it all down at the end. Console labels change, so follow the intent, not
the exact wording. Replace `REGION`, `ACCOUNT_ID` and `BUCKET` below.

## 1. Account and safety

1. Create the AWS account and turn on **MFA for the root user**. Then stop using root.
2. Create an admin user for yourself (IAM Identity Center/SSO or an IAM user) and use that.
3. Create a **budget alert** (AWS Budgets, monthly cost budget, e.g. 25 USD, email alerts at
   50%, 80% and 100%). Do this before creating any other resource.
4. Check your plan under Billing: new accounts get credits, and some services may be limited
   on the Free plan. If a service you need is blocked, switch to the Paid plan (credits still
   apply to eligible usage).

## 2. Local access

```bash
aws configure sso          # preferred; or `aws configure` with an IAM user's keys
aws sts get-caller-identity
```

Never commit access keys. `.env` and `~/.aws` are not in the repo.

Choose **one region** for everything and set `AWS_REGION` in `.env`. Check that Bedrock offers
your embedding model and a Claude model there.

## 3. Bedrock

Open the Bedrock console in your region and enable access to Titan Text Embeddings V2 and a
Claude model if the console asks for it. Then:

```bash
make check-bedrock
python -m tickettriage.check_bedrock --judge   # needs BEDROCK_JUDGE_MODEL_ID in .env
```

If the model id is rejected, the console shows the exact id or inference-profile id for your
region. Use that.

## 4. S3 bucket (model versions, snapshots)

- Name: `tickettriage-<yourname>-<random>` (bucket names are global).
- Region: your chosen region. **Block all public access**: on.
- Prefixes you will use later: `models/`, `snapshots/`.

## 5. ECR repository (container images)

- Create a private repository named `tickettriage`.
- Optional: add a lifecycle rule that keeps only the last 5 images (storage costs money).

## 6. EC2 instance role (create it now, launch the instance in week 2)

Create an IAM role for EC2 named `tickettriage-ec2` and attach this policy (tighten the
Bedrock resource to specific model/inference-profile ARNs once everything works):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ModelBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::BUCKET", "arn:aws:s3:::BUCKET/*"]
    },
    {
      "Sid": "EcrLogin",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "EcrPull",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "arn:aws:ecr:REGION:ACCOUNT_ID:repository/tickettriage"
    },
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "*"
    }
  ]
}
```

The Converse API uses the same `bedrock:InvokeModel` permission.

## 7. Write down what you created

| Resource | Name / id | Created on |
|---|---|---|
| Region | | |
| S3 bucket | | |
| ECR repository | | |
| EC2 role | `tickettriage-ec2` | |

## Cost habits

- Stop (not just idle) the EC2 instance at the end of every session once it exists.
- Do not create a NAT gateway. It bills per hour and the design does not need one.
- Delete unattached Elastic IPs and old volumes.
- At the end of the project, delete everything listed in the table above.
