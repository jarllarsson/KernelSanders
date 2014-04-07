#pragma once
#include <fstream>
#include <string>

bool write_file_binary (std::string const & filename, 
						char const * data, size_t const bytes)
{
	std::ofstream b_stream(filename.c_str(), 
		std::fstream::out | std::fstream::binary);
	if (b_stream)
	{
		b_stream.write(data, bytes);
		return (b_stream.good());
	}
	return false;
}
