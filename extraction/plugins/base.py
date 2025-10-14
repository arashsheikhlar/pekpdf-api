# SPDX-License-Identifier: AGPL-3.0-only

"""
Plugin system for extraction tools.

This module defines the base classes and registry for extraction plugins,
allowing for extensible extraction capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from ..models import ExtractionOptions, PluginMetadata


class ExtractorPlugin(ABC):
    """Abstract base class for extraction plugins."""
    
    def __init__(self):
        """Initialize the plugin."""
        self.metadata = self.get_metadata()
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            PluginMetadata object with plugin information
        """
        pass
    
    @abstractmethod
    def can_handle(self, document_type: str, options: ExtractionOptions) -> bool:
        """
        Check if this plugin can handle the given document type and options.
        
        Args:
            document_type: Type of document to process
            options: Extraction options
            
        Returns:
            True if this plugin can handle the request
        """
        pass
    
    @abstractmethod
    def extract(self, text: str, pages_text: List[str], options: ExtractionOptions) -> Dict[str, Any]:
        """
        Extract data using this plugin.
        
        Args:
            text: Combined text from all pages
            pages_text: List of text for each page
            options: Extraction options
            
        Returns:
            Dictionary with extracted data
        """
        pass
    
    def get_name(self) -> str:
        """Get plugin name."""
        return self.metadata.name
    
    def get_version(self) -> str:
        """Get plugin version."""
        return self.metadata.version
    
    def get_description(self) -> str:
        """Get plugin description."""
        return self.metadata.description
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported document types."""
        return self.metadata.supported_types
    
    def get_priority(self) -> int:
        """Get plugin priority (higher = more important)."""
        return self.metadata.priority


class PluginRegistry:
    """Registry for managing extraction plugins."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, ExtractorPlugin] = {}
        self._plugins_by_type: Dict[str, List[ExtractorPlugin]] = {}
    
    def register_plugin(self, plugin: ExtractorPlugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        name = plugin.get_name()
        self._plugins[name] = plugin
        
        # Index by supported types
        for doc_type in plugin.get_supported_types():
            if doc_type not in self._plugins_by_type:
                self._plugins_by_type[doc_type] = []
            self._plugins_by_type[doc_type].append(plugin)
        
        # Sort plugins by priority (highest first)
        for doc_type in self._plugins_by_type:
            self._plugins_by_type[doc_type].sort(key=lambda p: p.get_priority(), reverse=True)
    
    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin by name.
        
        Args:
            name: Name of plugin to unregister
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            del self._plugins[name]
            
            # Remove from type index
            for doc_type in plugin.get_supported_types():
                if doc_type in self._plugins_by_type:
                    self._plugins_by_type[doc_type] = [
                        p for p in self._plugins_by_type[doc_type] if p.get_name() != name
                    ]
    
    def get_plugin(self, name: str) -> Optional[ExtractorPlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def get_plugins_for_type(self, document_type: str) -> List[ExtractorPlugin]:
        """
        Get all plugins that support a document type.
        
        Args:
            document_type: Document type to get plugins for
            
        Returns:
            List of plugins sorted by priority
        """
        return self._plugins_by_type.get(document_type, [])
    
    def get_all_plugins(self) -> List[ExtractorPlugin]:
        """
        Get all registered plugins.
        
        Returns:
            List of all plugins
        """
        return list(self._plugins.values())
    
    def get_plugin_names(self) -> List[str]:
        """
        Get names of all registered plugins.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def find_applicable_plugins(self, document_type: str, options: ExtractionOptions) -> List[ExtractorPlugin]:
        """
        Find plugins that can handle the given document type and options.
        
        Args:
            document_type: Document type
            options: Extraction options
            
        Returns:
            List of applicable plugins sorted by priority
        """
        applicable = []
        for plugin in self.get_plugins_for_type(document_type):
            if plugin.can_handle(document_type, options):
                applicable.append(plugin)
        
        return applicable
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._plugins_by_type.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_plugins': len(self._plugins),
            'plugins_by_type': {doc_type: len(plugins) for doc_type, plugins in self._plugins_by_type.items()},
            'plugin_names': list(self._plugins.keys())
        }


# Global plugin registry instance
plugin_registry = PluginRegistry()
