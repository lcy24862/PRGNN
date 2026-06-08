import ants

fixed_template = 'TEMPLATE_FDGPET_100.nii'
clean_file = None # Your data here

fixed = ants.image_read(fixed_template)
moving = ants.image_read(clean_file)
reg = ants.registration(
    fixed=fixed, moving=moving,
    type_of_transform="SyN",
    random_seed=42,
    verbose=False,
)
warped = reg["warpedmovout"]
ants.image_write(warped, 'warped_output.nii')
