#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Paul Hofer paul.hofer@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""A plane section whose thickness is a random field: a seeded Gaussian random field (``gstools``),
evaluated at each element's centre and mapped to a thickness by an expression.

The field is a deterministic function of position for a given seed, whatever elements it is
evaluated for and in which order, so an element receives the same thickness whenever and wherever it
is created -- at setup, in any process of a domain-decomposed run, or when it moves to another
process.
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import (
    MaterialParameterFromFieldSchema,
    WriteMaterialPropertiesToFileSchema,
)
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.math import createFunction
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField


@dataclass(frozen=True)
class PlaneRandomThicknessSectionSchema:
    """The options of :class:`Section`."""

    thickness: float | None = schemaField(description="reference thickness", dtype=float, default=None, required=True)
    variance: float | None = schemaField(
        description="variance of the Gaussian random field", dtype=float, default=None, required=True
    )
    lengthScale: float | None = schemaField(
        description="length scale of the Gaussian random field", dtype=float, default=None, required=True
    )
    seed: int | None = schemaField(
        description="seed of the random field; the same seed gives the same field",
        dtype=int,
        default=None,
        required=True,
    )
    thicknessExpression: str = schemaField(
        description=(
            "the thickness from the element centre x, the reference thickness ref and the random field value "
            "rand at x"
        ),
        dtype=str,
        default="ref * (1.0 + rand)",
        optionName="f(x,ref,rand)",
    )
    materialParameterFromField: tuple[MaterialParameterFromFieldSchema, ...] = subKeywordField(
        description="use material properties given by an analytical field",
        schema=MaterialParameterFromFieldSchema,
    )
    writeMaterialPropertiesToFile: tuple[WriteMaterialPropertiesToFileSchema, ...] = subKeywordField(
        description="export material properties to file",
        schema=WriteMaterialPropertiesToFileSchema,
    )
    elementSets: str | None = datalineField(
        description="elementSets as comma separated list of element sets for this section", required=True
    )


class Section(PlaneSection):
    """A plane section with a random thickness per element; see the module documentation.

    Parameters
    ----------
    name
        The name of this section.
    model
        The model tree.
    material
        The material (or marmot material provider dict) assigned to this section.
    elementSets
        The element sets this section is applied to.
    configuration
        The options; see :class:`PlaneRandomThicknessSectionSchema`.
    """

    #: Option schema for this section, per OptionSchemaProvider.
    schema = PlaneRandomThicknessSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: PlaneRandomThicknessSectionSchema = PlaneRandomThicknessSectionSchema(),
    ):
        super().__init__(name, model, material, elementSets, configuration=configuration)

        # gstools is imported lazily: its Cython extension does not declare free-threading support,
        # so importing it re-enables the GIL process-wide and would disable thread-parallel
        # computations for ALL simulations, even those not using random fields.
        try:
            import gstools
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "the 'planeRandomThickness' section requires the 'gstools' package "
                "(install via 'pip install gstools' or 'mamba install -c conda-forge gstools')"
            ) from e

        covariance = gstools.Gaussian(
            dim=model.domainSize, var=configuration.variance, len_scale=configuration.lengthScale
        )
        #: The random field, a deterministic function of position for the given seed.
        self.randomField = gstools.SRF(covariance, seed=configuration.seed)
        self._thicknessExpression = createFunction(configuration.thicknessExpression, "x", "ref", "rand", model=model)

    def thicknessOf(self, element) -> float:
        """The thickness of an element: the expression of the reference thickness and the random
        field, both at the element's centre.

        Parameters
        ----------
        element
            The element.

        Returns
        -------
        float
            The thickness.
        """

        centre = np.asarray(element.getCoordinatesAtCenter(), dtype=float)
        return float(np.asarray(self._thicknessExpression(centre, self.thickness, self.randomField(centre)[0])).item())
