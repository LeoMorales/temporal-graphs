"""
.. module:: temporal_graph
   :platform: Unix, Windows
   :synopsis: attemp of Kostakos temporal graphs implementation in Python.

.. moduleauthor:: Leo Morales <moralesleonardo.rw@gmail.com>


"""

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
import functools
import time
import datetime
import pandas as pd
import re
from itertools import takewhile
import math


TIPOS_FECHAS = ["<class 'datetime.datetime'>",
                "<class 'pandas._libs.tslibs.timestamps.Timestamp'>",
                # Porque timestamp pelado me parecia confuso
                "<class 'numpy.datetime64'>"]

PALETA_MCDONALDS = {
    #'nodes_color': '#23512f',
    'nodes_color': '#1f4729',
    'links_color': '#c20d00',
    'temp_links_color': '#23512f',
    'label_color': '#ffce00'
}

EXP_NODO = re.compile(r"(?P<nodo>[a-zA-Z-_]+)(?P<posicion>[0-9]+)")
 
FORMATO_FECHA = '%d/%m/%Y'

class TemporalGraph:
    '''Grafo temporal

        _times (list):
            Los _times nos sirven para manejar las columnas (visual) de tiempos.
            Es una lista de datetime.datetime.
            Corresponde a todos los tiempos que entran en juego en el grafo

        _last_node_appearance (dict):
            Estructura para mantener el ultimo (mayor) datetime.datetime en el que un nodo de su correspondiente fila es utilizado.
            Por ejemplo:

                {

                    'a': datetime.datetime(2018, 12, 19, 14, 34, 14, 736048)

                    'b': datetime.datetime(2018, 12, 19, 14, 40, 34, 736048)

                    ...

                }

        _graph (networkx.classes.digraph.DiGraph):
            Grafo dirigido que le asigna peso 0.0 (instantáneo) a los links entre nodos de distintas filas y peso con diferencia en segundos entre nodos desagregados de una misma fila.

        _step (int):
            Paso que nos sirve para guardar una imagen (img_<step>) cada vez que se agrega un enlace/link.

        _node_labels (dict):
            Diccionario con las correspondencias entre los labels originales y los labels cortos ('a'. 'b', ...)
    '''

    def __init__(self, tiempos):
        '''Inicializa un grafo temporal.

        Args:
            tiempos (list): Lista de tiempos (datetime.datetime o similares).

        Raises:
            Exception: Si no se especifican los tiempos del grafo.
            
            Exception: Los tipos de la lista no se encuentran en:
                - datetime.datetime
                - pandas._libs.tslibs.timestamps.Timestamp
                - numpy.datetime64

        '''
        if len(tiempos) == 0:
            raise Exception(
                'Debe especificar los tiempos con los que trabaja el grafo')
        if str(type(tiempos[0])) not in TIPOS_FECHAS:
            exception_msg = '''Se esperan tiempos con formatos de fechas: {}\nSe encontró: {} ({})'''.format(
                ', '.join(TIPOS_FECHAS),
                str(type(tiempos[0])),
                type(tiempos[0]).__name__)
            raise Exception(exception_msg)
        self._times = sorted(list(set(tiempos)), reverse=False)
        self._last_node_appearance = {}
        self._graph = nx.DiGraph()
        self._step = 0
        self._node_labels = {}

    def _get_node_number(self, time, silent_fail=True):
        '''Retorna el numero de columna a la que corresponde el tiempo recibido (visualmente).

        Args:
            time (int): Un tiempo.
            
            silent_fail (bool): Indica si en caso de no encontrar el tiempo recibido, devolver por defecto 0.
        
        Raise:
            Exception: Si el tiempo especificado no existe en el grafo.
        '''
        try:
            return self._times.index(time) + 1
        except ValueError:
            if silent_fail:
                return 0
            else:
                exception_mes = '''El tiempo especificado ({})
                    no corresponde a un tiempo disponible en el grafo'''.format(time)
                raise Exception(exception_mes)

    def _get_representation(self, node_name, tiempo):
        '''Un nodo 'a' en un grafo sera desagregado y representado como 'aX' donde X es el lugar en el que se ubica _tiempo_ en la lista de todos los tiempos ordenados

        Args:
            node_name (str): Nombre del nodo en el grafo estatico. Por ej: 'a', 'x', etc.
                TODO: Poder recibir labels mas compuestos

            tiempo (datetime.datetime): tiempo en el que participa el nodo.

        Returns:
            str: En donde el node_name está concatenado con el numero que representa la ubicación del tiempo en la lista de tiempos ordenados ascendentemente.
        '''
        return '{}{}'.format(node_name.strip(), self._get_node_number(tiempo))

    def _update_last_appearances(self, nodeA, nodeB, time):
        '''Guarda la ultima aparicion del nodo (temporalmente hablando) en la estructura de ultimas apariciones.

        TODO: Analizar si aporta algo el guardar todas las apariciones de un nodo en una lista...

        Returns:
            None
        '''
        last_appearance = self._last_node_appearance.get(nodeA, None)
        if not last_appearance or (time > last_appearance):
            self._last_node_appearance[nodeA] = time

        last_appearance = self._last_node_appearance.get(nodeB, None)
        if not last_appearance or (time > last_appearance):
            self._last_node_appearance[nodeB] = time

    def create_link(self, sender, receiver, time, link_label='-'):
        '''Crea un link entre los nodos recibidos y ademas crea un link con linea punteada a la aparición anterior de la fila del nodo correspondiente.

        Args:
            sender (str): Nodo (estatico) desde el cual se comienza la interacción.

            receiver (str): Nodo (estatico) desde el cual se recibe la interacción.
                Precondicion: sender != receiver

            time (datetime.datetime): Tiempo en el que se produce la interacción.


        Returns:
            tuple: tupla con los elementos:
                - sender (str),
                - receiver (str),
                - time (datetime),
                - instancia creada del nodo origen (str)

        Raises:
            Exception: El ``time`` debe ser un datetime.datetime 
                de python.
        '''
        # limpiar sender y receiver:
        sender = sender.strip()
        receiver = receiver.strip()
        sender = re.sub('\d', 'X', sender)
        receiver = re.sub('\d', 'X', receiver)
        sender = sender.replace('.', '_')
        receiver = receiver.replace('.', '_')

        # - Comprobar el tipo de dato del tiempo recibido:
        if str(type(time)) != "<class 'datetime.datetime'>":
            try:
                time = datetime.datetime.strptime(str(time), '%d/%m/%Y')
            except:
                raise Exception(
                    'El atributo time debe ser de tipo "datetime.datetime" o un tipo casteable a datetime')

        # Crear el link entre los nodos
        # Instancias:
        instance_node_origin = self._get_representation(sender, time)
        instance_node_target = self._get_representation(receiver, time)

        self._graph.add_edge(instance_node_origin,
                             instance_node_target,
                             weight=0.0, label=link_label)

        # Luego, si corresponde, crear los links de los nodos que recibimos
        # y su anterior aparición en el grafo:

        # sender:
        last_time_appearance = self._last_node_appearance.get(sender, None)
        if last_time_appearance:  # Si es la 1ra vez que aparece, no hacer nada
            if time > last_time_appearance:
                # hay un link directo
                link_weight = (time - last_time_appearance).total_seconds()
                self._graph.add_edge(self._get_representation(sender, last_time_appearance),
                                     instance_node_origin,
                                     weight=link_weight)
            elif time < last_time_appearance:
                # debemos dividir el link que ya existía, porque
                # nos enviaron un nodo que va en el medio
                print('reacomodar links horizontales emisor')

        # receiver:
        last_time_appearance = self._last_node_appearance.get(receiver, None)
        if last_time_appearance:  # Si es la 1ra vez que aparece, no hacer nada
            if time > last_time_appearance:
                # hay un link directo
                link_weight = (time - last_time_appearance).total_seconds()
                self._graph.add_edge(self._get_representation(receiver, last_time_appearance),
                                     instance_node_target,
                                     weight=link_weight)
            elif time < last_time_appearance:
                # debemos dividir el link que ya existía, porque
                # nos enviaron un nodo que va en el medio
                print('reacomodar links horizontales receptor')

        self._update_last_appearances(sender, receiver, time)
        return (sender, receiver, time, instance_node_origin)

    def get_graph(self):
        ''' Retorna el grafo de networkx '''
        return self._graph

    def get_extended_labels(self):
        '''Retorna las correspondencias label original - label corto'''
        return self._node_labels

    def __ordena_letras_nodos(self, a, b):
        '''Sirve para ordenar los nodos del grafo por la parte alfabética de la etiqueta de los nodo.
        El orden es por tamaño y a igual tamaño, alfabeticamente

        Returns:
            int:
                -1 si a --> a >= b
                1  si b --> b < a
        '''
        if len(a) == len(b):
            return -1 if a <= b else 1
        else:
            return 1 if (len(a) > len(b)) else -1

    def _get_base_node(self, node):
        '''Retorna el nodo base para la instancia de nodo recibida:
        Por ejemplo: Para "ab34" (str) retorna "ab" (str)
        '''
        try:
            return EXP_NODO.match(node).groupdict().get('nodo')
        except AttributeError as e:
            # AttributeError: 'NoneType' object has no attribute 'groupdict'
            raise Exception(
                "AttributeError: 'NoneType' object has no attribute 'groupdict'\nNodo: {}".format(node))

    def _substract_position(self, node):
        '''Retorna la posicion en la que se ubica la instancia de nodo recibida:
        Por ejemplo: Para "ab34" retorna 34

        Returns:
            int: el tiempo para la instancia de nodo indicada.
        '''
        return int(
            EXP_NODO.match(node).groupdict().get('posicion'))

    def _temporal_graph_positions(self, subgraph=None):
        '''En base a los nodos del grafo, devuelve sus posiciones en un temporal graph.
        Calcular las posiciones de los nodos es de relevancia en la visualizacion del grafo temporal:
            - Instancias de los nodos en posicion horizontal.
            - Interacciones entre nodos en posicion vertical.

        Returns:
            dict: con las posiciones de cada nodo según el tiempo en el que participen.
                Cada valor del dict es un numpy.array con coords x e y por cada label del nodo.

                {
                    'a1': [0, 0],
                    'a2': [1, 0],
                    'a3': [2, 0],
                    'b1': [1, 1],
                    'b2': [2, 1],
                    ...
                }   
        '''
        subgraph = subgraph if subgraph else self._graph
        xbase = 0.5
        ybase = 0.5
        # Armar una grilla de columnas por los labels letras:
        all_letters = sorted(list(set(
            [self._get_base_node(node)
                for node
                in subgraph.nodes()])),
            key=functools.cmp_to_key(self.__ordena_letras_nodos))
        positions = {}
        for node in subgraph.nodes():
            ypos = ybase + all_letters.index(self._get_base_node(node))
            xpos = xbase + float(
                self._substract_position(node))
            #print('position:', xpos, ypos)
            positions[node] = np.array(
                (
                    xpos,
                    ypos * -1
                )
            )
        print('Positions', len(positions), 'nodos')
        return positions

    def _draw_interconnections(self, links, ax, grid_positions, graph, sg=None, link_color='k'):
        '''Dibuja todos los enlaces recibidos con una forma curva

        Args:
            links (list): Lista de tuplas con los labels de los nodos origen y destino.

            ax (matplotlib.axes.Axes): Ejes sobre los cuales se dibujan los enlaces.

        Returns:
        
        --> Codigo base de la funcion
            --> https://groups.google.com/forum/#!topic/networkx-discuss/FwYk0ixLDuY

        '''
        #grid_positions = self._temporal_graph_positions()
        # A cada nodo del grafo, se le asigna un circulo (geometría)
        graph = graph if graph else self._graph

        for nodo in graph:
            circulo_nodo = Circle(
                grid_positions[nodo], radius=0.05, alpha=0.06)
            ax.add_patch(circulo_nodo)
            graph.node[nodo]['patch'] = circulo_nodo

        for (origen, destino) in links:
            n1 = graph.node[origen]['patch']
            n2 = graph.node[destino]['patch']

            # setup enlace:
            rad = 0.2  # curvatura del enlace
            alpha = 0.65
            color = link_color
            link = FancyArrowPatch(
                n1.center,
                n2.center,
                patchA=n1,
                patchB=n2,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=%s' % rad,
                mutation_scale=10.0,
                lw=2,
                alpha=alpha,
                color=color
            )
            ax.add_patch(link)

        return

    def get_subgraph_by_label_condition(self, label_condition, show_instances=False):
        '''Retorna el subgrafo que resulta del filtrado de los enlaces que tienen label igual a label_condition.
        El tema que me surgió acá es que si filtramos los enlaces, hay instancias que quedan afuera y después deberíamos crear un enlace con linea punteada que sume los dos pesos de enlaces.

        Args:
            label_condition (str): Valor a comparar contra el atributo 'label'

            show_instances (bool): Indica si se muestran los enlaces horizontales (y los nodos) de los nodos que participan en los enlaces filtrados.

        Return:
            Digraph: Copia del grafo filtrado.
                Es copia para poder trabjar con el grafo, sin modificar el original.
        '''
        edges = [
            (emisor, receptor)
            for emisor, receptor, data
            in self._graph.edges(data=True)
            if ((data.get('label') == label_condition) or (show_instances and data.get('weight') > 0.0))]
        # Bug fixed?
        return self._graph.edge_subgraph(edges).copy()
    

    def plot(self, only_save=False, output_folder='output', paleta=PALETA_MCDONALDS,
        filter_labels=None):
        '''Dibuja el grafo temporal

        Args:
            only_save (bool): Indica si se deben guardar un png del grafo en lugar de mostrarlo una vez terminado por pantalla (True).
                TODO: Esto está pensado para generar el gif de forma manual. Será posible generarlo de forma automática?

            output_folder (str): Carpeta en la cual se van a guardar las imagenes generadas.
                Por defecto, intenta guardarlas en una carpeta 'output'.

            paleta (dict): Paleta de colores para el grafo.
                Debe contener las claves:
                    - 'nodes_color',
                    - 'links_color',
                    - 'temp_links_color'
                para indicar los colores de los nodos, de los links entre nodos distintos y los links entre nodos del mismo nodo base (a1, a2, a3, etc --> a) respectivamente.

        Returns:
            Digraph: Grafo que se utilizo para dibujar.

        '''
        # setups:
        line_width = 4
        nodes_size = 1000

        if only_save:
            # matplotlib.use('Agg')
            plt.ioff()

        # filtrar los nodos que interesen:
        if filter_labels:
            work_graph = self.get_subgraph_by_label_condition(
                filter_labels, show_instances=True)
        else:
            work_graph = self._graph

        # los arcos con peso != 0 son con linea punteada, los que no tienen peso, con linea continua:
        # En este modelo, los mensajes se consideran eventos instantaneos.
        econtin = [(u, v) for (u, v, d) in work_graph.edges(
            data=True) if d['weight'] == 0.0]
        edashed = [(u, v) for (u, v, d) in work_graph.edges(
            data=True) if d['weight'] > 0.0]

        # posiciones de cada nodo: en base al label, los arcos no importan en este paso:
        pos = self._temporal_graph_positions(work_graph)

        plt.figure(figsize=(20, 14))
        # edges
        ax = plt.gca()
        # nodes
        nx.draw_networkx_nodes(
            work_graph,
            pos,
            node_size=nodes_size,
            alpha=.65,
            node_color=paleta['nodes_color'],
            ax=ax)
        
        # labels
        nx.draw_networkx_labels(
            work_graph,
            pos,
            font_color=paleta['label_color'],
            font_size=14,
            font_family='sans-serif',
            ax=ax
            )

        
        # Dibujar conecciones entre nodos 'desagregados' (participaciones temporales del nodo):
        nx.draw_networkx_edges(
            work_graph,
            pos,
            edgelist=edashed,
            width=line_width,
            alpha=0.27,
            edge_color=paleta['temp_links_color'],
            ax=ax,
            style='dashed',
            arrows=False)

        # Dibujar conecciones entre nodos (interacciones):
        self._draw_interconnections(
            econtin,
            ax,
            pos,
            work_graph,
            link_color=paleta['links_color'])


        plt.axis('off')

        if only_save:
            plt.savefig('{}/img_{}.png'.format(output_folder, self._step))
            self._step += 1
            plt.close()
        else:
            plt.show()

        return work_graph

    def _build_links(self, data_row,
                     column_sender,
                     column_destination,
                     column_time,
                     column_label=None,
                     verbose=True,
                     save_steps_images=False):
        '''Crea enlaces en el grafo temporal según la data recibida.

        Args:
            data_row (pandas.core.series.Series): Informacion de el o los enlaces

            column_sender (str): Indice en la serie 'data_row' que corresponde al emisor.
                Puede contener múltiples emisores separados por coma.

            column_destination (str): Indice en la serie que corresponde al receptor.
                Puede contener múltiples receptores separados por coma.

            column_time (str): Indice en la serie que corresponde al tiempo en el que se produce la interacción.

            verbose (bool): Indica si se imprime un mensajito por cada enlace creado.

            save_steps_images (bool): Indica si se plotea todo el grafo cada vez que se crea un nuevo enlace.

        '''
        link_label = data_row[column_label].strip() if column_label else '-'
        for origin in data_row[column_sender].split(','):
            origin = origin.strip()
            for destination in data_row[column_destination].split(','):
                destination = destination.strip()
                try:
                    # Cuando pandas lee una fecha en un csv, las interpreta con su
                    # propio tipo de datos, que podemos convertir a datetime.datetime
                    # de python con el metodo to_pydatetime:
                    time = data_row[column_time].to_pydatetime()
                except:
                    raise Exception(
                        'El atributo no se puede convertir a datetime.datetime')
                origin, destination, time, instance_origin = self.create_link(
                    origin, destination, time, link_label)
                if verbose:
                    print('Enlace: ', origin, destination,
                          data_row[column_time], instance_origin)
                if save_steps_images:
                    self.plot(only_save=True)

    def build_links_from_data(self, data,
                              col_sender='sender',
                              col_destination='recipient',
                              col_time='time',
                              col_label=None,
                              save_images=False, verbose=True):
        '''Crea links en base al dataframe con la data correspondiente.

        Args:

            data (pandas.Dataframe):
            ::
                | sender | recipient | time
                |  (str) |  (str)    | (datetime.datetime)
                --------------------------------------------------
                | A      |   B       | 2018-12-19 14:34:14.736048
                | A      |   C, E    | 2018-12-19 14:34:15.736424
                --------------------------------------------------
            ::

            column_sender (str): Columna que corresponde al emisor.

            column_destination (str): Columna que corresponde al receptor.

            column_time (str): Columna que corresponde al tiempo en el que se produce la interacción.

            save_images (bool): Indica si se tiene que guardar el grafo cada vez que se agrega un nuevo enlace.
        '''
        if not {col_sender, col_destination, col_time}.issubset(set(data.columns.values)):
            exception_msg = '''
            El dataframe especificado debe contener las columnas:
            {}, {}, {}'''.format(col_sender,
                                 col_destination,
                                 col_time)
            raise Exception(exception_msg)

        # Ordenar por tiempos:
        data = data.sort_values(by=[col_time])
        build_link = functools.partial(
            self._build_links,
            column_sender=col_sender,
            column_destination=col_destination,
            column_time=col_time,
            column_label=col_label,
            save_steps_images=save_images,
            verbose=verbose)
        # aplicar la funcion anterior parcialmente evaluada a cada fila.
        data.apply(build_link, axis=1)

    def _get_node_instances(self, node_base, from_time=None, to_time=None):
        '''Devuelve las instancias en el tiempo para el nodo base recibido.

        Args:
            node_base (str): Nodo del cual se desean obtener las instancias.

            from_time (int): Indica que se desean obtener instancias que se encuentren en tiempos superiores (o igual) a este valor.

            to_time (int): Indica que se desean obtener instancias que se encuentren en tiempos inferiores (o igual) a este valor.

        Returns:
            list: Instancias del nodo recibido.
                Lista vacia si no hay instancias para el nodo. 
        '''
        instances = filter(
            lambda n: self._get_base_node(n) == node_base,
            self._graph.nodes()
        )

        if from_time:
            instances = filter(
                lambda n: self._substract_position(n) >= from_time,
                instances)
        if to_time:
            instances = filter(
                lambda n: self._substract_position(n) <= to_time,
                instances)

        return list(instances)

    def _get_node_participations(self, node):
        '''Retorna los tiempos en los que el nodo recibido tiene una instancia.

        Args:
            node (str): Nodo. Por ejemplo: 'a', 'b', 'c', etc.

        Returns:
            list of int: Tiempos de las instancias del nodo recibido.
        '''
        participations = [
            self._substract_position(node_i)
            for node_i
            in self._get_node_instances(node)
        ]
        return sorted(participations)

    def _get_first_instance_after_time(self, node_base, time=None):
        '''Devuelve la instancia del nodo en el tiempo especificado si existe, sino el mas próximo a partir del tiempo recibido.
        Sino se especifica el tiempo, se devuelve la primera instancia que se encuentra.

        Args:
            node_base (str): Nodo base del que se desea encontrar la instancia.

            time (int): Tiempo a partir del cual se desea encontrar la primer instancia para el nodo base.

        Raises:
            Exception si a partir del valor de tiempo recibido no es posible encontrar una instancia del nodo base.
        '''
        searched_node = '{}{}'.format(node_base, time)
        if self._graph.has_node(searched_node):
            return searched_node

        # El nodo no tiene una instancia en el tiempo recibido,
        # obtenemos todos los tiempos en donde exiten instancias:
        participations = self._get_node_participations(node_base)

        # si no se especificó el tiempo, devolver la primera aparición:
        if not time:
            return '{}{}'.format(node_base, participations[0])

        # caso contrario, devolver la mas próxima en el tiempo:
        for n in participations:
            if n > time:
                return '{}{}'.format(node_base, n)
        else:
            raise Exception(
                'No se puede encontrar una instancia del nodo "{}" posterior al tiempo {} recibido'.format(
                    node_base, time))

    def _get_last_instance_before_time(self, node_base, time_max=None):
        '''Retorna la instancia del nodo recibido en el tiempo recibido, o la mas próxima anterior (sería comenzando de derecha a izquierda).
        Si no se especifica el tiempo max, devuelve el último nodo instancia que se encuentre para el node_base.

        Args:
            node_base (str): Nodo

            time_max (int): El tiempo hasta el cual se busca la instancia del nodo.
        '''
        participations = self._get_node_participations(node_base)
        if not time_max:
            return '{}{}'.format(node_base, participations[-1])

        if time_max in participations:
            return '{}{}'.format(node_base, time_max)
        else:
            return '{}{}'.format(
                node_base,
                list(takewhile(lambda v: v < time_max, participations))[-1]
            )


    def temporal_proximity(self, node_from, node_to, time_from=None, time_to=None, verbose=False):
        '''Devuelve la proximidad temporal entre los nodos

        Args:
            node_from (str): Label del nodo (base) desde el cual se calcula la proximidad temporal. Por ej: 'A', 'B', etc.

            node_to (str): Label del nodo (base) hasta el cual se calcula la proximidad temporal. Por ej: 'A', 'B', etc.

            time_from (int): precondicion temporal (tiempo desde)

            time_to (int): poscondicion temporal (tiempo hasta)

        Returns:
            list: Lista de los nodos que representan el camino mas corto en cuanto a lo temporal, desde node_from hasta node_to.
        '''
        self.__logging(
            verbose,
            'Temporal proximity from {} to {}\n'.format(
                node_from,
                node_to)
        )
        paths = []
        if time_from or ((not time_from) and time_to):
            # Si tenemos tiempo desde --> p(A,D,ti,null) o
            # si tenemos solo tiempo hasta --> p(A,D,null,ti)
            # el camino se calcula desde la 1ra instancia del nodo encontrada:
            origin_instances = [
                self._get_first_instance_after_time(node_from, time_from)]
            self.__logging(verbose,
                           '\tSearching from: {}'.format(origin_instances[0]))
        else:
            # En los otros dos casos:
            # p(A,D,ti,tj) y p(A,D,null,null)
            # buscar desde todas las instancias del nodo from:
            origin_instances = self._get_node_instances(
                node_from, from_time=time_from, to_time=time_to)
            self.__logging(verbose,
                           '\tSearching from all origin node instances {}'.format(
                               node_from))

        for origin_instance in origin_instances:
            self.__logging(verbose,
                           '\tFrom: {}'.format(origin_instance))
            origin_time = self._substract_position(origin_instance)
            if not time_to:
                destination_instances = self._get_node_instances(
                    node_to, from_time=origin_time)
            else:
                try:
                    destination_instances = [
                        self._get_last_instance_before_time(node_to, time_to)]
                except Exception as e:
                    raise Exception('El nodo destino {} no tiene instancias anteriores al tiempo {}'.format(
                        node_to,
                        time_to))

            for destination_instance in destination_instances:
                self.__logging(verbose,
                               '\tDestination: {}?'.format(destination_instance))
                if nx.algorithms.has_path(self._graph, origin_instance, destination_instance):
                    # Si hay un camino entre las instancias:
                    # guardar
                    path_found = nx.algorithms.shortest_path(
                        self._graph,
                        origin_instance,
                        destination_instance)
                    paths.append(path_found)
                    self.__logging(verbose,
                                   '\t\tHave path, save: \n\t\t{}\n\t\tnext node instance destination...'.format(path_found))
                    # cortar:
                    break
        path = min(paths, key=lambda path: self.weight(path)) if paths else []

        return path

    def weight(self, path):
        '''Retorna el peso del camino recibido.

        Args:
            path (list): lista de nodos del camino.
                Se espera que la lista contenga instancias de nodos.

        Returns:
            float: Peso del camino.
        '''
        if not path:
            return 0
        from_node = path[0]
        acum = 0
        for node in path[1:]:
            try:
                acum += self._graph.get_edge_data(from_node,
                                                  node).get('weight')
            except Exception as e:
                # AttributeError: 'NoneType' object has no attribute 'get'
                raise Exception('No hay enlace desde {} a {}'.format(
                    from_node, node))
            from_node = node
        return acum

    def _node_in_graph(self, node):
        '''Retorna si el nodo (no la instancia) se encuentra en el grafo'''
        return node in self._last_node_appearance

    def average_temporal_proximity(self, node_from, node_to, verbose=False):
        '''ATP

        Args:
            node_from (str): Label del nodo (base) desde el cual se calcula la proximidad temporal promedio.
                Por ej: 'A', 'B', etc.

            node_to (str): Label del nodo (base) hasta el cual se calcula la proximidad temporal promedio.
                Por ej: 'A', 'B', etc.

            verbose (bool): Indica si se muestra la salida de los pasos realizados.

        Returns:
            float: En promedio, cuánto tiempo toma ir desde X hasta Y.
        '''
        if not self._node_in_graph(node_from):
            print('Node from not in graph')
            return
        if not self._node_in_graph(node_to):
            print('Node to not in graph')
            return

        self.__logging(
            verbose,
            'Average temporal proximity from {} to {}\n'.format(
                node_from,
                node_to)
        )

        acum = 0.0
        paths = []
        instances_origin = self._get_node_instances(node_from)
        self.__logging(verbose, '-- Iteration start --')
        for origin_instance in instances_origin:
            origin_time = self._substract_position(origin_instance)
            temp_proximity = self.temporal_proximity(
                node_from,
                node_to,
                time_from=origin_time,
                verbose=verbose)
            paths.append(temp_proximity)
            if verbose:
                print('In t:{}\n\tdistance from {} to {}\n\tpath: {}\n\n'.format(
                    origin_time, node_from, node_to, temp_proximity))

        self.__logging(verbose, '-- Iteration finished --')
        # Eliminar paths vacios:
        paths = list(filter(lambda path: len(path) != 0, paths))
        # Obtener los pesos de los paths:
        paths_weights = list(map(self.weight, paths))
        try:
            return sum(paths_weights) / len(paths_weights)
        except ZeroDivisionError as e:
            return None

    def __logging(self, verbose_mode, msg):
        '''Funcion para ahorrarme la comprobación del verbose mode en 
        las funciones del módulo'''
        if verbose_mode:
            print(msg)

    def _get_nodes(self):
        '''Retorna los nodos del grafo (no las instancias)

        Old way:
            return sorted(list(set(
                [
                    self._get_base_node(node_i)
                    for node_i
                    in self._graph.nodes()
                ])))
            Evita recorrer todas las instancias del grafo, ordenando
            las claves del diccionario de apariciones de nodos.
        '''
        return sorted(self._last_node_appearance)

    def average_temporal_proximity_from_node(self, node):
        '''Retorna las proximidades temporales promedio del nodo
        hacia el resto de los nodos

        Args:
            node (str): Nodo desde el cual calcular las
                proximidades temporales promedio.
                Por ejemplo: 'A'

        Returns:
            dict: Diccionario con las proximidades temporales promedio 
                desde el nodo recibido. Por ejemplo, para 'A' se 
                puede devolver:

                {
                    'A': 0.0,

                    'B': 561600.0,

                    'C': 43200.0,

                    'D': 144000.0,

                    'E': 43200.0
                
                }
        '''
        avg_tmp_proxs = {}
        for node_y in self._get_nodes():
            avg_tmp_proxs[node_y] = self.average_temporal_proximity(
                node,
                node_y)
        return avg_tmp_proxs

    def average_temporal_reach(self, node):
        '''On average, how quickly does X reach the rest of the network.

        Lease: ```P out```

        Args:
            node (str): Nodo. Por ejemplo: 'A'.

        Returns:
            float, o None si desde el nodo no se alcanza ningun otro nodo.
        '''
        avg_tmp_proxs_from_node = self.average_temporal_proximity_from_node(
            node)
        # eliminar valores nulos:
        avg_tmp_proxs_from_node = list(filter(
            lambda proximity: proximity is not None,
            avg_tmp_proxs_from_node.values()))
        try:
            # restar 1 para no contar la prox temp promedio hacien el propio nodo:
            return sum(avg_tmp_proxs_from_node) / (len(avg_tmp_proxs_from_node) - 1)
        except Exception as e:
            return None

    def average_temporal_proximity_to_node(self, node):
        '''Retorna las proximidades temporales promedio del resto de los nodos del grafo hacia el nodo recibido.

        Args:
            node (str): Nodo hacia el cual calcular las proximidades temporales promedio.
                Por ejemplo: 'D'

        Returns:
            dict: Diccionario con las proximidades temporales promedio desde el nodo recibido.

                Por ejemplo, para 'D' se puede devolver:

                {

                    'A': 144000.0,

                    'B': 374400.0,

                    'C': None,

                    'D': 0.0,

                    'E': 86400.0

                }
        '''
        avg_tmp_proxs = {}
        for node_x in self._get_nodes():
            avg_tmp_proxs[node_x] = self.average_temporal_proximity(
                node_x,
                node)
        return avg_tmp_proxs

    def average_temporal_reachability(self, node):
        '''On average, how quickly is X reached by the rest of the network.

        Lease: ```P in```

        Args:
            node (str): Nodo. Por ejemplo: 'A'.

        Returns:
            float, o None si el nodo no es alcanzado por ningun otro nodo.
        '''
        avg_tmp_proxs_to_node = self.average_temporal_proximity_to_node(node)
        avg_tmp_proxs_to_node = list(filter(
            lambda proximity: proximity is not None,
            avg_tmp_proxs_to_node.values()))

        try:
            # restar 1 para no contar la prox temp promedio hacien el propio nodo:
            return sum(avg_tmp_proxs_to_node) / (len(avg_tmp_proxs_to_node) - 1)
        except Exception as e:
            return None

    def _clean_data(self, df, col_sender, col_recipient, col_time, format_times=True,
        formato_fecha=FORMATO_FECHA, col_label=None):
        '''Construye un dataframe listo-para-ser-entrada-del-grafo-temporal.
        A partir de un dataframe existente que tenga columnas con tipos que coincidan con los tipos del dataframe de entrada --> (str, str, fecha)

        Debemos indicar cual de las columnas del dataframe de entrada debe ser tratada como sender, recipient y time, así hacemos las transformaciones correspondientes
        Todos las etiquetas se convierten en etiquetas acortadas: 'a', 'b', [...],'z', 'aa', 'ab', [...], 'az' y así...
        La correspondencias entre estos alias y los labels originales quedan almacenados en self._node_labels

        Args:
            df (pandas.DataFrame): Dataframe con la informacion cruda.

            col_sender (str): Cual de las columnas del dataframe
                especifican el emisor del mensaje

            col_recipient (str): Cual de las columnas del dataframe
                especifica el receptor de cada mensaje.

            col_time (str): Cual de las columnas debe utilizarse
                como fechas de tiempo en el que sucedió el evento.
                Esta columna (generalmente luego de ser levantada
                desde un csv) es de tipo str (object en pandas) y
                tiene el formato dd/mm/YYYY.

            format_time (boolean): Indica si la columna de tiempo es un string y se deben formatear.
                Si el df tiene la columna con tiempos datetime, se indica que se ignore este paso (False)

            formato_fecha (str): Formato a partir del cual se construye la fecha (el datetime) del evento.

        '''
        # Create alphabet list of lowercase letters
        nodos = df[col_sender].append(df[col_recipient]).unique()
        alphabet = self.__create_alphabet(
            nodos.size)
        # alphabet.pop()  --> 'a'
        # alphabet.pop()  --> 'b'
        self._node_labels = {}

        for nodo in nodos:
            self._node_labels[nodo] = alphabet.pop()

        data_cleaned = {
            'sender': df[col_sender].apply(lambda emisor: self._node_labels[emisor]),
            'recipient': df[col_recipient].apply(lambda receptor: self._node_labels[receptor]),
        }
        if format_times:
            data_cleaned['time'] = df[col_time].apply(
                lambda fecha: datetime.datetime.strptime(fecha, formato_fecha))

        if col_label:
            data_cleaned['label'] = df[col_label]            
        
        return pd.DataFrame(data_cleaned)

    def __create_alphabet(self, num):
        '''Crea y retorna el alfabeto para representar con letras tanta cantidad de elementos como lo indique ``num``

        Retorna la lista dada vuelta para poder ir usando .pop() y mantener el orden alfabético.

        '''
        alphabet = []
        base = ''
        counter = 0
        while len(alphabet) <= num:
            for letter in range(97, 123):
                alphabet.append(base+chr(letter))
            base = alphabet[counter]
            counter += 1
        return alphabet[::-1]

    def clean_and_build_links_data(
        self, data,
        col_sender, col_recipient, col_time,
        col_label=None, verbose=False):
        ''' Limpiar y crear enlaces
        '''
        self.build_links_from_data(
            self._clean_data(
                data, col_sender, col_recipient, col_time, col_label=col_label),
            col_label='label',
            verbose=verbose)



###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Module utils:

def __create_alphabet(num):
    '''Crea y retorna el alfabeto para representar con letras tanta cantidad de elementos como lo indique ``num``

    Retorna la lista dada vuelta para poder ir usando .pop() y mantener el orden alfabético.

    '''
    alphabet = []
    base = ''
    counter = 0
    while len(alphabet) <= num:
        for letter in range(97, 123):
            alphabet.append(base+chr(letter))
        base = alphabet[counter]
        counter += 1
    return alphabet[::-1]
    # return alphabet


def limpiar_data(df, col_sender, col_recipient, col_time, formato_fecha=FORMATO_FECHA):
    '''Construye un dataframe listo-para-ser-entrada-del-grafo-temporal, a partir
    de un dataframe existente que tenga columnas con tipos que coincidan con los
    tipos del dataframe de entrada --> (str, str, fecha)

    Debemos indicar cual de las columnas del dataframe de entrada
    debe ser tratada como sender, recipient y time, así hacemos las
    transformaciones correspondientes

    Args:
        col_sender (str): Cual de las columnas del dataframe
            especifican el emisor del mensaje

        col_recipient (str): Cual de las columnas del dataframe
            especifica el receptor de cada mensaje.

        col_time (str): Cual de las columnas debe utilizarse
            como fechas de tiempo en el que sucedió el evento.
            Esta columna (generalmente luego de ser levantada
            desde un csv) es de tipo str (object en pandas) y
            tiene el formato dd/mm/YYYY.

    '''
    # Create alphabet list of lowercase letters
    # TODO: Si son mas de 27 nodos distintos? OK!
    nodos = df[col_sender].append(df[col_recipient]).unique()
    alphabet = __create_alphabet(
        nodos.size)
    # alphabet.pop()  --> 'a'
    # alphabet.pop()  --> 'b'
    reemplazos = {}
    # for emisor in df[col_sender].unique():
    #     if not emisor in reemplazos:
    #         reemplazos[emisor] = alphabet.pop()
    # for receptor in df[col_recipient].unique():
    #     if not receptor in reemplazos:
    #         reemplazos[receptor] = alphabet.pop()

    for nodo in nodos:
        reemplazos[nodo] = alphabet.pop()

    return reemplazos, pd.DataFrame({
        'sender': df[col_sender].apply(lambda emisor: reemplazos[emisor]),
        'recipient': df[col_recipient].apply(lambda receptor: reemplazos[receptor]),
        'time': df[col_time].apply(lambda fecha: datetime.datetime.strptime(fecha, formato_fecha)),
    })
