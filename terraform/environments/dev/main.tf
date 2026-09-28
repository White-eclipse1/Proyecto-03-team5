locals {
  name = "mlops-p2-dev"
}

module "network" {
  source = "../../modules/network"

  name               = local.name
  cidr_block         = "10.10.0.0/16"
  availability_zones = var.availability_zones
}

module "compute" {
  source = "../../modules/compute"

  name      = local.name
  vpc_id    = module.network.vpc_id
  vpc_cidr  = module.network.cidr_block
  subnet_id = module.network.subnet_ids[0]
  ami_id    = var.ami_id
}

module "data" {
  source = "../../modules/data"

  name                      = local.name
  vpc_id                    = module.network.vpc_id
  subnet_ids                = module.network.subnet_ids
  compute_security_group_id = module.compute.security_group_id
}

module "storage" {
  source = "../../modules/storage"

  name = local.name
}
