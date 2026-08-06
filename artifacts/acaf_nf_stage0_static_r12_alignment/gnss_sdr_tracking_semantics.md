# GNSS-SDR tracking semantics

**source:** Parent manifest + receiver.conf identify the GNSS-SDR tracking dump; field names are read from MAT files.

**fields:** {'PRN': 'GPS PRN', 'PRN_start_sample_count': 'raw complex-sample index at integration start', 'aux1': 'tracker auxiliary/remnant value; only converted with aux*code_freq/fs when candidate semantics require it', 'carrier_doppler_hz': 'stored carrier Doppler estimate', 'code_freq_chips': 'stored code NCO rate', 'Prompt_I_Q': 'stored prompt correlator components'}

**interval_contract:** Intervals are true adjacent rows within a channel/PRN: [previous_sample_count,sample_count) or [sample_count,next_sample_count), never a fixed 25,000-sample assumption.

**cross_prn_overlap:** Explicitly allowed: tracker channels/PRNs may have same-epoch temporal overlap. Only cross-role temporal overlap is prohibited.
