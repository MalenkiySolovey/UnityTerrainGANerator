using UnityEngine;
using UnityTerrainGANerator.Preview;
using UnityTerrainGANerator.TerrainGeneration;

namespace UnityTerrainGANerator.Demo
{
    public sealed class ExampleSceneController : MonoBehaviour
    {
        public TerrainGANPreviewProfile previewProfile;
        public TerrainGenerator terrainGenerator;
        public Texture2D defaultBiomeTexture;
        public bool applyProfileBeforeGeneration = true;

        public void GenerateDefault()
        {
            if (applyProfileBeforeGeneration)
            {
                ApplyPreviewProfile();
            }

            Texture2D texture = previewProfile != null && previewProfile.biomeTexture != null
                ? previewProfile.biomeTexture
                : defaultBiomeTexture;
            GenerateFromTexture(texture);
        }

        public void GenerateFromTexture(Texture2D biomeTexture)
        {
            if (terrainGenerator == null)
            {
                terrainGenerator = GetComponent<TerrainGenerator>();
            }

            if (terrainGenerator == null)
            {
                throw new System.InvalidOperationException("TerrainGenerator is not assigned.");
            }

            if (biomeTexture == null)
            {
                throw new System.ArgumentNullException(nameof(biomeTexture));
            }

            terrainGenerator.GenerateFromBiomeTexture(biomeTexture);
        }

        public void ApplyPreviewProfile()
        {
            if (previewProfile == null)
            {
                return;
            }

            if (terrainGenerator == null)
            {
                terrainGenerator = GetComponent<TerrainGenerator>();
            }

            previewProfile.ApplyTo(terrainGenerator);
            defaultBiomeTexture = previewProfile.biomeTexture;
        }
    }
}
