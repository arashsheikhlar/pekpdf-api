# SPDX-License-Identifier: AGPL-3.0-only

"""
Plugin initialization and registration.

This module handles automatic discovery and registration of extraction plugins.
"""

from .base import ExtractorPlugin, PluginRegistry, plugin_registry
from .table_extractor import TableExtractorPlugin
from .formula_extractor import FormulaExtractorPlugin


def register_builtin_plugins() -> None:
    """Register all built-in plugins."""
    # Register table extractor plugin
    table_plugin = TableExtractorPlugin()
    plugin_registry.register_plugin(table_plugin)
    
    # Register formula extractor plugin
    formula_plugin = FormulaExtractorPlugin()
    plugin_registry.register_plugin(formula_plugin)


def discover_plugins() -> None:
    """
    Discover and register plugins from the plugin directory.
    
    This function can be extended to automatically discover plugins
    from external modules or directories.
    """
    # For now, just register built-in plugins
    register_builtin_plugins()
    
    # TODO: Add dynamic plugin discovery from external modules
    # This could involve:
    # 1. Scanning a plugin directory for Python modules
    # 2. Loading modules and looking for ExtractorPlugin subclasses
    # 3. Instantiating and registering found plugins


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry.
    
    Returns:
        The global plugin registry instance
    """
    return plugin_registry


def get_available_plugins() -> list[ExtractorPlugin]:
    """
    Get all available plugins.
    
    Returns:
        List of all registered plugins
    """
    return plugin_registry.get_all_plugins()


def get_plugins_for_document_type(document_type: str) -> list[ExtractorPlugin]:
    """
    Get plugins that support a specific document type.
    
    Args:
        document_type: Document type to get plugins for
        
    Returns:
        List of plugins that support the document type
    """
    return plugin_registry.get_plugins_for_type(document_type)


# Auto-discover plugins on module import
discover_plugins()


# Export main classes and functions
__all__ = [
    'ExtractorPlugin',
    'PluginRegistry', 
    'plugin_registry',
    'TableExtractorPlugin',
    'FormulaExtractorPlugin',
    'register_builtin_plugins',
    'discover_plugins',
    'get_plugin_registry',
    'get_available_plugins',
    'get_plugins_for_document_type'
]
