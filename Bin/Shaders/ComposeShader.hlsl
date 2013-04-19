Texture2D g_diffuse				: register(t0);
Texture2D g_normal				: register(t1);

Texture2D g_depth				: register(t10);

SamplerState g_samplerPointWrap : register(s0);

struct VertexIn
{
	float3 position : POSITION;
};
struct VertexOut
{
    float4 position	: SV_POSITION;
};

VertexOut VS(VertexIn p_input)
{
	VertexOut vout;
	vout.position = float4(p_input.position,1.0f);
	vout.texCoord = p_input.texCoord;
    
	return vout;
}

float4 PS(VertexOut input) : SV_TARGET
{
	uint3 index;
	index.xy = input.position.xy;
	index.z = 0;

	float4 finalCol = g_diffuse.Load( index);
	
	// test
	finalCol.rgb = float3(0.0f,0.0f,1.0f);
	
	return float4( finalCol.rgb, 1.0f );
}