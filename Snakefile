rule all:
	input:
		[f"output_{index}" for index in range(1,11)]

rule run_script:
	group:  "simulations"
	output:
		directory("output_{index}")
	conda:
		"environment.yaml"
	threads: 1
	resources:
		mem_mb=500,
		disk_mb=100
	shell:
		"""
		python ./bin/script.py --sim_num {wildcards.index} --output_dir {output}
		"""