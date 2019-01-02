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

TIPOS_FECHAS = ["<class 'datetime.datetime'>",
               "<class 'pandas._libs.tslibs.timestamps.Timestamp'>"]  # Porque timestamp pelado me parecia confuso

PALETA_MCDONALDS = {
    'nodes_color': '#ffce00',
    'links_color': '#c20d00',
    'temp_links_color': '#23512f',
    'label_color': '#23512f'
}
class TemporalGraph:
    ''' Grafo temporal
        
        _times (list): Los _times nos sirven para manejar las
            columnas (visual) de tiempos.
            Lista de datetime.datetime.
            Son todos los tiempos que entran en juego en el grafo
        
        _last_node_appearance (dict): Estructura para mantener
            el ultimo (mayor) datetime.datetime en el que un nodo
            de su correspondiente fila es utilizado.
            Por ejemplo:
                {
                    'a': datetime.datetime(2018, 12, 19, 14, 34, 14, 736048)
                    'b': datetime.datetime(2018, 12, 19, 14, 40, 34, 736048)
                    ...
                }
        
        _graph (networkx.classes.digraph.DiGraph): Grafo dirigido
            que le asigna peso 0.0 (instantáneo) a los links entre
            nodos de distintas filas y peso con diferencia en segundos
            entre nodos desagregados de una misma fila.
            
        _step (int): Paso que nos sirve para guardar una imagen (img_<step>)
            cada vez que se agrega un enlace/link.
    '''
    def __init__(self, tiempos):
        if len(tiempos) == 0:
            raise Exception('Debe especificar los tiempos con los que trabaja el grafo')
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
        
    def _get_node_number(self, tiempo, silent_fail=True):
        ''' Retorna el numero de columna a la que corresponde
            el tiempo recibido (visualmente).
        '''
        try:
            return self._times.index(tiempo) +1
        except ValueError:
            if silent_fail:
                return 0
            else:
                exception_mes = '''El tiempo especificado ({})
                    no corresponde a un tiempo disponible en el grafo'''.format(tiempo)
                raise Exception(exception_mes)  
    
    def _get_representation(self, node_name, tiempo):
        ''' Un nodo 'a' en un grafo sera desagregado y representado
            como 'aX' donde X es el lugar en el que se ubica _tiempo_
            en la lista de todos los tiempos ordenados
            
            Args:
                node_name (str): Nombre del nodo en el grafo
                    estatico. Por ej: 'a', 'x', etc.
                    TODO: Poder recibir labels mas compuestos
                tiempo (datetime.datetime): tiempo en la
                    que participa el nodo.
            
            Return:
                str: En donde el node_name está concatenado
                    con el numero que representa la ubicación
                    del tiempo en la lista de tiempos ordenados
                    ascendentemente.
        '''
        return '{}{}'.format(node_name.strip(), self._get_node_number(tiempo))
    
    def _update_last_appearances(self, nodeA, nodeB, time):
        ''' Guarda la ultima aparicion del nodo (temporalmente hablando)
            en la estructura de ultimas apariciones.
            
            TODO: Analizar si aporta algo el guardar todas las apariciones
            de un nodo en una lista...
        '''
        last_appearance = self._last_node_appearance.get(nodeA, None)
        if not last_appearance or (time > last_appearance):
            self._last_node_appearance[nodeA] = time
        
        last_appearance = self._last_node_appearance.get(nodeB, None)
        if not last_appearance or (time > last_appearance):
            self._last_node_appearance[nodeB] = time
        
    def create_link(self, sender, receiver, time):
        ''' Debemos crear un link entre los nodos recibidos y ademas
            crear un link con linea punteada a la aparición anterior
            de la fila del nodo correspondiente.
            
            Args:
                sender (str): Nodo (estatico) desde el cual se comienza
                    la interacción.
                receiver (str): Nodo (estatico) desde el cual se recibe
                    la interacción.
                time (datetime.datetime): Tiempo en el que se produce
                    la interacción.
            
            Precondicion:
                sender != receiver
            
            Return:
                None
                
            Raises:
                Exception: El ``time`` debe ser un datetime.datetime
                    de python.
        '''
        sender = sender.strip()
        receiver = receiver.strip()
        if str(type(time)) != "<class 'datetime.datetime'>":
            try:
                time = datetime.datetime.strptime(str(time), '%d/%m/%Y')
            except:
                raise Exception('El atributo time debe ser de tipo "datetime.datetime" o un tipo casteable a datetime')
        
        # Crear el link entre los nodos
        self._graph.add_edge(self._get_representation(sender, time),
                             self._get_representation(receiver, time),
                             weight=0.0)
        
        # Luego, si corresponde, crear los links de los nodos que recibimos
        # y su anterior aparición en el grafo:
        
        # sender:
        last_time_appearance = self._last_node_appearance.get(sender, None)
        if last_time_appearance:  # Si es la 1ra vez que aparece, no hacer nada
            if time > last_time_appearance:
                # hay un link directo
                link_weight = (time - last_time_appearance).total_seconds()
                self._graph.add_edge(self._get_representation(sender, last_time_appearance),
                                     self._get_representation(sender, time),
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
                                     self._get_representation(receiver, time),
                                     weight=link_weight)
            elif time < last_time_appearance:
                # debemos dividir el link que ya existía, porque
                # nos enviaron un nodo que va en el medio
                print('reacomodar links horizontales receptor')
            
        self._update_last_appearances(sender, receiver, time)

    def get_graph(self):
        return self._graph
    
    def __ordena_letras_nodos(self, a, b):
        ''' Sirve para ordenar los nodos del grafo por la parte
            alfabética de la etiqueta de los nodo.
            El orden es por tamaño y a igual tamaño, alfabeticamente

            Returns:
                -1 si a --> a >= b
                1  si b --> b < a
    
        '''
        if len(a) == len(b):
            return -1 if a <= b else 1
        else:
            return 1 if (len(a) > len(b)) else -1
    
    def _temporal_graph_positions(self):
        ''' Dame los nodos que te devuelvo sus posiciones en un temporal graph
            Returns:
                dict con las posiciones de cada nodo según el tiempo
                    en el que participen. Cada valor del dict es un
                    numpy.array con coords x e y por cada label del nodo
                {
                    'a1': [0, 0],
                    'a2': [1, 0],
                    'a3': [2, 0],
                    'b1': [1, 1],
                    'b2': [2, 1],
                    ...
                }   
        '''
        xbase = 0.0
        ybase = 0.0
        # Armar una grilla de columnas por los labels letras:
        exp = re.compile(r"(?P<letra>[a-zA-Z]+)(?P<numero>[0-9]+)")
        all_letters = sorted(list(set( 
            [ exp.match(node).groupdict().get('letra')
                for node
                in self._graph.nodes()] )),
            key=functools.cmp_to_key(self.__ordena_letras_nodos))
        positions = {}
        for node in self._graph.nodes():
            ypos = ybase - all_letters.index(
                exp.match(node).groupdict().get('letra'))
            xpos = xbase + float(
                exp.match(node).groupdict().get('numero'))
            positions[node] = np.array((xpos, ypos))
        return positions

    def _draw_interconnections(self, links, ax, sg=None, link_color='k'):
        ''' Dibuja todos los enlaces recibidos con una forma curva
            Args:
                links (list): Lista de tuplas con los labels
                    de los nodos origen y destino
                ax (matplotlib.axes.Axes)

            Return:
                El ultimo enlace.
            
            --> Codigo base de la funcion
                --> https://groups.google.com/forum/#!topic/networkx-discuss/FwYk0ixLDuY

        '''
        grid_positions = self._temporal_graph_positions()
        # A cada nodo del grafo, se le asigna un circulo (geometría)
        for nodo in self._graph:
            circulo_nodo = Circle( grid_positions[nodo], radius=0.15, alpha=0.01 )
            ax.add_patch(circulo_nodo)
            self._graph.node[nodo]['patch'] = circulo_nodo

        for ( origen, destino ) in links:
            n1 = self._graph.node[origen]['patch']
            n2 = self._graph.node[destino]['patch']

            # setup enlace:
            rad = 0.2  # curvatura del enlace
            alpha = 0.65
            color = link_color
            link = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                                   arrowstyle='-|>',
                                   connectionstyle='arc3,rad=%s' % rad,
                                   mutation_scale=10.0,
                                   lw=2,
                                   alpha=alpha,
                                   color=color)
            ax.add_patch(link)

        return

    def plot(self, only_save=False, output_folder='output', paleta=PALETA_MCDONALDS):
        ''' Dibuja el grafo temporal

            Args:
                only_save (bool): Indica si se deben guardar un png del grafo en
                    lugar de mostrarlo una vez terminado por pantalla (True).
                    TODO: Esto está pensado para generar el gif de forma manual.
                        Será posible generarlo de forma automática?

                output_folder (str): Carpeta en la cual se van a guardar las
                    imagenes generadas. Por defecto, intenta guardarlas
                    en una carpeta 'output'

                paleta (dict): Paleta de colores para el grafo.
                    Debe contener las claves 'nodes_color', 'links_color',
                    'temp_links_color' para indicar los colores de los nodos,
                    de los links entre nodos distintos y los links entre nodos
                    del mismo nodo base (a1, a2, a3, etc --> a) respectivamente.


        '''
        # setups:
        line_width  = 4
        nodes_color = paleta['nodes_color']
        nodes_size = 1000
        dashed_edges_color = paleta['temp_links_color']
        continuos_edges_color = paleta['links_color']
        
        if only_save:
            ##matplotlib.use('Agg')
            plt.ioff()

        # los arcos con peso != 0 son con linea punteada, los que no tienen peso, con linea continua:
        # En este modelo, los mensajes se consideran eventos instantaneos.
        econtin = [(u, v) for (u, v, d) in self._graph.edges(data=True) if d['weight'] == 0.0]
        edashed = [(u, v) for (u, v, d) in self._graph.edges(data=True) if d['weight'] > 0.0]

        # posiciones de cada nodo: en base al label, los arcos no importan en este paso:
        pos = self._temporal_graph_positions()
        
        plt.figure(figsize=(18, 12))
        # nodes
        nx.draw_networkx_nodes(self._graph, pos, node_size=nodes_size, alpha=.45,
                              node_color=nodes_color)
        # edges
        ax=plt.gca()

        # Dibujar conecciones entre nodos (interacciones):
        self._draw_interconnections(econtin, ax,
                                    link_color=continuos_edges_color)
        # Dibujar conecciones entre nodos 'desagregados' (participaciones temporales del nodo):
        nx.draw_networkx_edges(self._graph, pos, edgelist=edashed,
                               width=line_width, alpha=0.25,
                               edge_color=dashed_edges_color, style='dashed',
                               arrows=False, ax=ax)

        # labels
        nx.draw_networkx_labels(
            self._graph, pos,
            font_color=paleta['label_color'],
            font_size=14,
            font_family='sans-serif')

        plt.axis('off')
        
        if only_save:
            plt.savefig('{}/img_{}.png'.format(output_folder, self._step))
            self._step += 1
            plt.close()
        else:
            plt.show()
    
    def _build_links(self, data_row,
                     column_sender,
                     column_destination,
                     column_time,
                     verbose=True,
                     save_steps_images=False):
        ''' Crea enlaces en el grafo temporal según la data recibida.
            
            Args:
                data_row (pandas.core.series.Series): Informacion de el o los enlaces
                column_sender (str): Indice en la serie que corresponde al emisor. Puede
                    contener múltiples emisores separados por coma.
                column_destination (str): Indice en la serie que corresponde al receptor.
                    Puede contener múltiples receptores separados por coma.
                column_time (str): Indice en la serie que corresponde al tiempo en el 
                    que se produce la interacción.
                verbose (bool): Indica si se imprime un mensajito por cada enlace creado,
                save_steps_images (bool): Indica si se plotea todo el grafo cada vez
                    que se crea un nuevo enlace. 
            
        '''
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
                    raise Exception('El atributo no se puede convertir a datetime.datetime')
                self.create_link(origin, destination, time)
                if verbose:
                    print('Enlace: ', origin, destination, data_row[column_time])
                if save_steps_images:
                    self.plot(only_save=True)
    
    def build_links_from_data(self, data,
                            col_sender='sender',
                            col_destination='recipient',
                            col_time='time',
                            save_images=False, verbose=True):
        ''' Crea links en base al dataframe con la data correspondiente.
            Args:
            
                data (pandas.Dataframe): 
                    | sender | recipient | time
                    |  str   |  str      | datetime.datetime
                    --------------------------------------------------
                    | A      |   B       | 2018-12-19 14:34:14.736048
                    | A      |   C, E    | 2018-12-19 14:34:15.736424
                    --------------------------------------------------
                
                column_sender (str): Columna que corresponde al emisor.
                
                column_destination (str): Columna que corresponde al receptor.
                
                column_time (str): Columna que corresponde al tiempo en el 
                    que se produce la interacción.
                    
                save_images (bool): Indica si se tiene que guardar el grafo
                    cada vez que se agrega un nuevo enlace.
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
        build_link = functools.partial(self._build_links,
                                       column_sender=col_sender,
                                       column_destination=col_destination,
                                       column_time=col_time,
                                       save_steps_images=save_images,
                                       verbose=verbose)
        data.apply(build_link, axis=1)


def __create_alphabet(num):
    alphabet = []
    base = ''
    counter = 0
    while len(alphabet) <= num:
        for letter in range(97,123):
            alphabet.append(base+chr(letter))
        base = alphabet[counter]
        counter += 1
    return alphabet[::-1]
    #return alphabet
    
def limpiar_data(df, col_sender, col_recipient, col_time, formato_fecha='%d/%m/%Y'):
    ''' Construye un dataframe listo-para-ser-entrada-del-grafo-temporal, a partir
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
    alphabet = __create_alphabet(df[col_sender].append(df[col_recipient]).unique().size)
    #alphabet.pop()  --> 'a'
    #alphabet.pop()  --> 'b'
    reemplazos = {}
    for emisor in df[col_sender].unique():
        if not emisor in reemplazos:
            reemplazos[emisor] = alphabet.pop()
    for receptor in df[col_recipient].unique():
        if not receptor in reemplazos:
            reemplazos[receptor] = alphabet.pop()
    
    return reemplazos, pd.DataFrame({
        'sender': df[col_sender].apply(lambda emisor: reemplazos[emisor]),
        'recipient': df[col_recipient].apply(lambda receptor: reemplazos[receptor]),
        'time': df[col_time].apply(lambda fecha: datetime.datetime.strptime(fecha, formato_fecha))
    })


    
    
        