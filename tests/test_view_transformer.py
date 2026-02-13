"""
Unit tests for ViewTransformer module.

Author: Malav Patel
"""

import numpy as np


class TestViewTransformer:
    """Tests for ViewTransformer class."""

    def test_initialization_defaults(self):
        from src.view_transformer.view_transformer import ViewTransformer

        transformer = ViewTransformer()
        assert transformer.court_width == 68.0
        assert transformer.court_length == 23.32
        assert transformer.perspective_transformer is not None

    def test_initialization_custom_vertices(self):
        from src.view_transformer.view_transformer import ViewTransformer

        vertices = np.array([[0, 100], [0, 0], [100, 0], [100, 100]], dtype=np.float32)
        transformer = ViewTransformer(pixel_vertices=vertices)
        assert np.array_equal(transformer.pixel_vertices, vertices)

    def test_transform_point_inside(self):
        from src.view_transformer.view_transformer import ViewTransformer

        transformer = ViewTransformer()
        # Point inside the default polygon
        point = np.array([500, 500])
        result = transformer.transform_point(point)
        assert result is not None
        assert result.shape == (1, 2)

    def test_transform_point_outside(self):
        from src.view_transformer.view_transformer import ViewTransformer

        transformer = ViewTransformer()
        # Point clearly outside the court polygon
        point = np.array([1900, 50])
        result = transformer.transform_point(point)
        assert result is None

    def test_add_transformed_position_to_tracks(self):
        from src.view_transformer.view_transformer import ViewTransformer

        transformer = ViewTransformer()
        tracks = {
            "players": [
                {
                    1: {
                        "bbox": [400, 400, 500, 600],
                        "position_adjusted": (450, 600),
                    }
                }
            ]
        }

        transformer.add_transformed_position_to_tracks(tracks)
        assert "position_transformed" in tracks["players"][0][1]

    def test_add_transformed_handles_none_position(self):
        from src.view_transformer.view_transformer import ViewTransformer

        transformer = ViewTransformer()
        tracks = {"players": [{1: {"bbox": [0, 0, 10, 10], "position_adjusted": None}}]}

        transformer.add_transformed_position_to_tracks(tracks)
        assert tracks["players"][0][1]["position_transformed"] is None
