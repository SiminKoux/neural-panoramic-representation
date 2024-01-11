import torch

def sample_exclude_first_row(N, H, W, samples):
    exclude_indices = torch.cat([torch.arange(i*H*W, i*H*W + W) for i in range(N)])
    all_indices = torch.arange(H * W * N)
    valid_indices = torch.tensor([index for index in all_indices if index.item() not in exclude_indices])
    sampled_indices = valid_indices[torch.randperm(valid_indices.size(0))[:samples]]
    sampled_indices = sampled_indices.unsqueeze(-1)
    return sampled_indices

def sample_points_from_middle_rows(N, H, W, samples):
    # Determine the start and end indices for the middle rows
    middle_rows = H // 2
    start_index = (H - middle_rows) // 2
    # print("start_index:", start_index)
    end_index = start_index + middle_rows
    # print("end_index:", end_index)

    # Determine the flattened indices for the middle rows for random frames
    flat_indices = []
    for _ in range(samples):
      random_frame = torch.randint(0, N, (1,)).item()
      start_flat_index = start_index * W + random_frame * H * W
      end_flat_index = end_index * W + random_frame * H * W
      flat_indices.append(torch.randint(start_flat_index, end_flat_index, (1,)))

    flat_indices = torch.cat(flat_indices)     # [samples]
    flat_indices = flat_indices.unsqueeze(-1)  # [samples, 1]

    return flat_indices


def inverse_transform_sampling_indices(cdf_values, n_samples):
    """Inverse transform sampling to get indices."""
    rand_nums = torch.rand(n_samples)
    indices = torch.zeros(n_samples, dtype=torch.long)  # ensure the indices as long type
    
    for i, r in enumerate(rand_nums):
        candidate_indices = torch.where(cdf_values > r)[0]
        indices[i] = candidate_indices[0]  # Take the first index from the list of indices where cdf_values > r
    
    return indices


def non_uniform_sample_points(N, H, W, n_samples):
    # Spatial Sampling
    lat_values = torch.linspace(-torch.pi/2, torch.pi/2, H)  # H values between -π/2 and π/2
    pdf_values = torch.cos(lat_values)
    pdf_values = pdf_values / torch.sum(pdf_values)
    cdf_values = torch.cumsum(pdf_values, dim=0)
    sampled_latitude_indices = inverse_transform_sampling_indices(cdf_values, n_samples)
    sampled_longitudes = (2 * torch.pi * torch.rand(n_samples) - torch.pi).long()

    # Temporal Sampling
    sampled_frame_indices = torch.randint(0, N, (n_samples,)).long()

    # Convert the sampled indices into a tensor index for the [3, H*W*N] representation
    flat_indices = sampled_frame_indices * H * W + sampled_latitude_indices * W + sampled_longitudes
    flat_indices = flat_indices.unsqueeze(-1)  # [samples, 1]
    
    return flat_indices
